
class Sampler:
    def __init__(self, init_num, config) -> None:
        self.config = config
        self.init_num = init_num
        
    def sample(self, search_space, n_points = None, metadata = None, metadata_info = None):
        raise NotImplementedError("Sample method should be implemented by subclasses.")
    
    
    def check_metadata_avaliable(self, metadata):
        if metadata is None:
            return False
        return True