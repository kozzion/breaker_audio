class FilterBase:
    
    def __init__(self) -> None:
        pass
    
    def _check_input(self, array_input, array_output):
        if not len(array_input.shape) == 2:
            raise Exception('array_input does not have 2 dimensions')

        if not len(array_output.shape) == 2:
            raise Exception('array_output does not have 2 dimensions')

        if array_input.shape[0] != array_output.shape[0]:
            raise Exception('array_input size ' + str(array_input.shape) + ' does not match array_output ' + str(array_output.shape))

        if array_input.shape[1] != array_output.shape[1]:
            raise Exception('array_input size ' + str(array_input.shape) + ' does not match array_output ' + str(array_output.shape))
