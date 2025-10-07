
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR



def create_warmup(warmup_epochs,optimizer,start_factor= 0.01 , end_factor = 1):
    sched_warmup = LinearLR(
        optimizer=optimizer,
        start_factor=0.01,
        end_factor=1,
        total_iters=warmup_epochs

    )
    return sched_warmup

def create_cosine(T_max,optimizer,eta_min):
    sched_cosine = CosineAnnealingLR(
        optimizer=optimizer,
        T_max= T_max,
        eta_min= eta_min
    )
    return sched_cosine



