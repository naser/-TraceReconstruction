import sys
sys.path.insert(0, 'd:/Naser/3')
import ctf_reader
import bisect

trace_dir = 'd:/Naser/3/elk-extracted/elktracelong/kernel'
event_map, stream_hdrs = ctf_reader.parse_metadata(trace_dir + '/metadata')

# Check specific event IDs
for eid in [313, 65534, 20873, 2048]:
    k0, k1 = (0, eid), (1, eid)
    r0 = event_map.get(k0, 'MISSING')
    r1 = event_map.get(k1, 'MISSING')
    print(f'  (0,{eid}): {r0 if isinstance(r0, str) else r0[0] if r0 else "MISSING"}')
    print(f'  (1,{eid}): {r1 if isinstance(r1, str) else r1[0] if r1 else "MISSING"}')

# Print all event IDs in sorted order, looking for nearby IDs around 313
ids = sorted(k[1] for k in event_map if k[0]==0)
idx = bisect.bisect_left(ids, 310)
print('IDs near 313 (stream 0):', ids[max(0,idx-3):idx+5])
print('Total stream 0 events:', len(ids))
ids1 = sorted(k[1] for k in event_map if k[0]==1)
print('Total stream 1 events:', len(ids1))
print('Max stream 0 ID:', max(ids))
print('Max stream 1 ID:', max(ids1) if ids1 else 'none')
