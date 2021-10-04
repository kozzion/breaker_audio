class FilterBase:
    
    def __init__(self) -> None:
        pass
    
    def _check_input(self, array_input, array_output):
        if 1 < len(array_input.shape):
            raise Exception('array_input has to many dimensions')

        if 1 < len(array_output.shape):
            raise Exception('array_output has to many dimensions')

        if array_input.shape[0] != array_output.shape[0]:
            raise Exception('array_input size ' + str(array_input.shape) + ' does not match array_output ' + str(array_output.shape))
