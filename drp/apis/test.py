import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
# torch.multiprocessing.set_sharing_strategy('file_system')

def single_gpu_test(model,
                    data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    # model.eval()
    # results = {'output': torch.Tensor(), 'labels': torch.Tensor()}
    # dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    # for data in data_loader:
    #     with torch.no_grad():
    #         result = model(
    #             test_mode=True,
    #             **data)
    #     results['output'] = torch.cat((results['output'], result['output']), 0)
    #     results['labels'] = torch.cat((results['labels'], result['labels']), 0)
    #
    #     # get batch size
    #     for _, v in data.items():
    #         if isinstance(v, torch.Tensor):
    #             batch_size = v.size(0)
    #             break
    #     for _ in range(batch_size):
    #         prog_bar.update()
    # results['output'] = results['output'].numpy().flatten()
    # results['labels'] = results['labels'].numpy().flatten()
    # return results
    torch.multiprocessing.set_sharing_strategy('file_system')
    model.eval()
    
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(
                test_mode=True,
                **data)
        results.append(result)

        # get batch size
        batch_size = data_loader.batch_size
        for _, v in data.items():
            if isinstance(v, torch.Tensor):
                batch_size = v.size(0)
                break
        for _ in range(batch_size):
            prog_bar.update()
    return results
