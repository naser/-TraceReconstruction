# Pure Python LTTng CTF reader — no babeltrace dependency required.
# Reads LTTng kernel CTF traces and extracts syscall event names.
#
# Correctly handles:
#   - event_header_compact (stream 1): 5+27 bit header (4 bytes compact, 13 bytes extended)
#   - event_header_large   (stream 0): 16+32 bit header (6 bytes compact, 14 bytes extended)
#   - stream 0 event context: 24 bytes (_procname[16], _pid, _tid)
#   - All field types: fixed-size integers and null-terminated strings
#   - Correct packet context layout (includes packet_seq_num field)
#
# Usage:
#   python ctf_reader.py <trace_dir> [output_file]

import os
import re
import struct
import sys

# ---------------------------------------------------------------------------
# Constants for this LTTng kernel trace format (from TSDL metadata)
# ---------------------------------------------------------------------------

CTF_MAGIC = 0xC1FC1FC1

# Packet header: magic(4) + uuid(16) + stream_id(4) + stream_instance_id(8) = 32 bytes
PKT_HDR_SIZE = 32

# Packet context: timestamp_begin(8) + timestamp_end(8) + content_size(8) +
#                 packet_size(8) + packet_seq_num(8) + events_discarded(8) + cpu_id(4) = 52 bytes
PKT_CTX_SIZE = 52
# Offsets within packet context (relative to start of context)
PKT_CTX_CONTENT_SIZE_OFF = 16   # bytes, uint64 in bits
PKT_CTX_PACKET_SIZE_OFF  = 24   # bytes, uint64 in bits

# Stream 0 event context: _procname[16](8-bit×16) + _pid(32-bit) + _tid(32-bit) = 24 bytes
STREAM0_EVT_CTX_SIZE = 24

# ---------------------------------------------------------------------------
# 1. Parse metadata TSDL → event map
# ---------------------------------------------------------------------------

def _parse_event_fields(fields_text):
    """
    Parse the text inside 'fields := struct { ... }' and return a list of
    (kind, bits) tuples, where kind is 'int' or 'str' and bits is the total
    bit count for integers (always a multiple of 8) or None for strings.
    """
    fields = []
    pos = 0
    n = len(fields_text)
    while pos < n:
        # Skip whitespace
        while pos < n and fields_text[pos] in ' \t\n\r':
            pos += 1
        if pos >= n:
            break
        if fields_text[pos:pos+7] == 'integer':
            # Find the closing }
            brace = fields_text.index('{', pos)
            end_brace = fields_text.index('}', brace)
            int_spec = fields_text[brace:end_brace+1]
            # Find field name and optional [count]
            semi = fields_text.index(';', end_brace)
            name_part = fields_text[end_brace+1:semi]
            size_m = re.search(r'size\s*=\s*(\d+)', int_spec)
            arr_m  = re.search(r'\[(\d+)\]', name_part)
            elem_bits = int(size_m.group(1)) if size_m else 8
            count     = int(arr_m.group(1)) if arr_m else 1
            fields.append(('int', elem_bits * count))
            pos = semi + 1
        elif fields_text[pos:pos+6] == 'string':
            semi = fields_text.index(';', pos)
            fields.append(('str', None))
            pos = semi + 1
        else:
            pos += 1  # skip unknown tokens
    return fields


_META_MAGIC = 0x75D11D57   # LTTng CTF metadata stream magic
_META_HDR   = 37           # metadata packet header size in bytes
#  layout: magic(4) + uuid(16) + checksum(4) + content_size(4) + packet_size(4)
#        + compression(1) + encryption(1) + checksum_scheme(1) + major(1) + minor(1)
#  content_size and packet_size are in BYTES (uint32)


def _extract_tsdl(meta_path):
    """
    Extract pure TSDL text from a CTF metadata file by stripping the
    per-packet binary headers.  Falls back to plain text read if the
    file doesn't start with the metadata magic.
    """
    import struct as _struct
    with open(meta_path, "rb") as f:
        raw = f.read()

    if len(raw) < 4 or _struct.unpack_from("<I", raw, 0)[0] != _META_MAGIC:
        # Not a CTF metadata stream — treat as plain text
        return raw.decode("utf-8", errors="replace")

    chunks = []
    pos = 0
    while pos + _META_HDR <= len(raw):
        magic = _struct.unpack_from("<I", raw, pos)[0]
        if magic != _META_MAGIC:
            break
        content_bits = _struct.unpack_from("<I", raw, pos + 24)[0]  # uint32, in bits
        packet_bits  = _struct.unpack_from("<I", raw, pos + 28)[0]  # uint32, in bits
        content_bytes = content_bits >> 3
        packet_bytes  = packet_bits  >> 3
        text_start = pos + _META_HDR
        text_end   = pos + content_bytes
        if text_end > len(raw):
            text_end = len(raw)
        chunks.append(raw[text_start:text_end])
        pos += packet_bytes if packet_bytes > _META_HDR else (content_bytes if content_bytes > _META_HDR else 4096)

    return b"".join(chunks).decode("utf-8", errors="replace")


def parse_metadata(meta_path):
    """
    Parse the TSDL metadata file.
    Returns event_map: {(stream_id, event_id): (name, fields)}
    where fields is list of (kind, bits_or_None).
    Also returns stream_map: {stream_id: header_type}
    where header_type is 'compact' or 'large'.
    """
    text = _extract_tsdl(meta_path)

    event_map = {}

    # Extract event blocks (may have nested {} in field struct)
    pos = 0
    while True:
        idx = text.find('\nevent {', pos)
        if idx == -1:
            break
        # Walk forward to find the closing } at depth 1
        depth = 0
        end = idx
        for i in range(idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        event_text = text[idx:end+2]

        name_m   = re.search(r'name\s*=\s*"([^"]+)"', event_text)
        id_m     = re.search(r'(?<!\w)id\s*=\s*(\d+)', event_text)
        stream_m = re.search(r'stream_id\s*=\s*(\d+)', event_text)

        if name_m and id_m and stream_m:
            eid = int(id_m.group(1))
            sid = int(stream_m.group(1))
            name = name_m.group(1)
            # Extract fields struct content
            fm = re.search(r'fields\s*:=\s*struct\s*\{(.*)\}', event_text, re.DOTALL)
            fields = _parse_event_fields(fm.group(1)) if fm else []
            event_map[(sid, eid)] = (name, fields)

        pos = idx + 1

    # Parse stream header types from stream definitions
    stream_hdrs = {}
    for sm in re.finditer(r'stream\s*\{([^}]+)\}', text, re.DOTALL):
        sc = sm.group(1)
        sid_m = re.search(r'(?<!\w)id\s*=\s*(\d+)', sc)
        hdr_m = re.search(r'event\.header\s*:=\s*struct\s+(\w+)', sc)
        if sid_m and hdr_m:
            stype = 'large' if 'large' in hdr_m.group(1) else 'compact'
            stream_hdrs[int(sid_m.group(1))] = stype

    print(f"  Metadata: {len(event_map)} event types, streams={stream_hdrs}", file=sys.stderr)
    return event_map, stream_hdrs


# ---------------------------------------------------------------------------
# 2. Channel file parser
# ---------------------------------------------------------------------------

def _skip_fields(data, pos, fields):
    """Advance pos past all fields in a field list. Returns new pos."""
    for kind, bits in fields:
        if kind == 'str':
            # null-terminated string — scan for \x00
            while pos < len(data) and data[pos] != 0:
                pos += 1
            pos += 1  # consume the null terminator
        else:
            pos += bits >> 3  # bits is always multiple of 8
    return pos


def _read_event_header_compact(data, pos):
    """
    Read event_header_compact (stream 1).
    5-bit id + 27-bit ts = 4 bytes compact, OR
    5 bits=31 + 3 pad + 32-bit id + 64-bit ts = 13 bytes extended.
    Returns (event_id, new_pos).
    """
    b0 = data[pos]
    id5 = b0 & 0x1F
    if id5 != 31:
        return id5, pos + 4       # compact: 4 bytes
    else:
        ext_id = struct.unpack_from('<I', data, pos + 1)[0]
        return ext_id, pos + 13   # extended: 13 bytes


def _read_event_header_large(data, pos):
    """
    Read event_header_large (stream 0).
    16-bit id + 32-bit ts = 6 bytes compact, OR
    16 bits=0xFFFF + 32-bit id + 64-bit ts = 14 bytes extended.
    Returns (event_id, new_pos).
    """
    id16 = struct.unpack_from('<H', data, pos)[0]
    if id16 != 0xFFFF:
        return id16, pos + 6      # compact: 6 bytes
    else:
        ext_id = struct.unpack_from('<I', data, pos + 2)[0]
        return ext_id, pos + 14   # extended: 14 bytes


def extract_events_from_channel(channel_path, event_map, stream_hdrs):
    """
    Parse a CTF channel binary file and yield (event_name, stream_id) pairs
    for all syscall events.
    """
    with open(channel_path, 'rb') as f:
        raw = bytearray(f.read())

    n = len(raw)
    pos = 0
    packet_count = 0

    while pos + PKT_HDR_SIZE + PKT_CTX_SIZE <= n:
        # --- Verify packet magic ---
        if struct.unpack_from('<I', raw, pos)[0] != CTF_MAGIC:
            nxt = raw.find(struct.pack('<I', CTF_MAGIC), pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue

        pkt_start = pos
        packet_count += 1

        # --- Packet header (32 bytes) ---
        # magic(4) | uuid(16) | stream_id(4) | stream_instance_id(8)
        pkt_stream_id = struct.unpack_from('<I', raw, pos + 20)[0]
        pos += PKT_HDR_SIZE

        # --- Packet context (52 bytes) ---
        content_bits = struct.unpack_from('<Q', raw, pos + PKT_CTX_CONTENT_SIZE_OFF)[0]
        packet_bits  = struct.unpack_from('<Q', raw, pos + PKT_CTX_PACKET_SIZE_OFF)[0]
        pos += PKT_CTX_SIZE

        if not content_bits:
            pos = pkt_start + (packet_bits >> 3 if packet_bits else 4096)
            continue

        content_end = pkt_start + (content_bits >> 3)
        pkt_end     = pkt_start + (packet_bits >> 3 if packet_bits else content_bits >> 3)
        if content_end > n:
            content_end = n

        # Determine event header type for this stream
        hdr_type = stream_hdrs.get(pkt_stream_id, 'compact')
        is_stream0 = (pkt_stream_id == 0)

        # --- Parse events in this packet ---
        while pos < content_end:
            if pos + 4 > n:
                break
            try:
                if hdr_type == 'large':
                    event_id, pos = _read_event_header_large(raw, pos)
                else:
                    event_id, pos = _read_event_header_compact(raw, pos)
            except (struct.error, IndexError):
                break

            # Skip stream 0 event context: _procname[16] + _pid + _tid = 24 bytes
            if is_stream0:
                pos += STREAM0_EVT_CTX_SIZE

            # Look up event definition
            key = (pkt_stream_id, event_id)
            entry = event_map.get(key)
            if entry is None:
                # Lost sync — jump to end of packet
                pos = content_end
                break

            name, fields = entry
            if fields:
                pos = _skip_fields(raw, pos, fields)

            if 'syscall' in name:
                clean = (name
                         .replace('syscall_entry_', '')
                         .replace('syscall_exit_',  '')
                         .replace('compat_syscall_entry_', '')
                         .replace('compat_syscall_exit_',  ''))
                yield clean

        pos = pkt_end

    print(f"  {packet_count} packets  ← {os.path.basename(channel_path)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------------------------------

def read_trace(trace_dir, out_file=None):
    meta_path = os.path.join(trace_dir, "metadata")
    if not os.path.exists(meta_path):
        print(f"ERROR: no metadata in {trace_dir}", file=sys.stderr)
        return 0

    event_map, stream_hdrs = parse_metadata(meta_path)

    # Accept any binary channel file (not metadata, not index files)
    channel_files = sorted(
        f for f in os.listdir(trace_dir)
        if os.path.isfile(os.path.join(trace_dir, f))
        and f != 'metadata'
        and not f.endswith('.idx')
        and not f.startswith('.')
    )
    print(f"  Channel files ({len(channel_files)}): {channel_files}", file=sys.stderr)

    out = open(out_file, "w") if out_file else sys.stdout
    total = 0
    try:
        for ch in channel_files:
            ch_path = os.path.join(trace_dir, ch)
            for name in extract_events_from_channel(ch_path, event_map, stream_hdrs):
                out.write(name + "\n")
                total += 1
                if total % 500_000 == 0:
                    print(f"  {total:,} events written...", file=sys.stderr, flush=True)
    finally:
        if out_file:
            out.close()

    print(f"  Total syscall events: {total:,}", file=sys.stderr)
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ctf_reader.py <trace_dir> [output_file]")
        sys.exit(1)
    trace_dir = sys.argv[1]
    out_file  = sys.argv[2] if len(sys.argv) > 2 else None
    read_trace(trace_dir, out_file)
