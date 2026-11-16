import sys, wandb
for rid in sys.argv[1:]:
    r = wandb.init(project="hack_RL", id=rid, resume="must", settings=wandb.Settings(console="off"))
    r.summary["stopped_by_user"] = True
    r.finish(exit_code=1)
    print("closed", rid)
