from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
import sys
e=E(sys.argv[1],size_guidance={"scalars":0}); e.Reload()
for t in sorted(e.Tags()["scalars"]):
    s=e.Scalars(t); print(t, len(s), s[0].step, s[-1].step)
