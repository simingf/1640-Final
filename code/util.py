import inspect

def dbg(*torch_tensors):
    frame = inspect.currentframe().f_back
    for torch_tensor in torch_tensors:
        for var_name, var_value in frame.f_locals.items():
            if var_value is torch_tensor:
                print(f"{var_name}: {torch_tensor.shape}, {torch_tensor.dtype}")
                break