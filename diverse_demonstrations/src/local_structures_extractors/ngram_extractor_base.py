from abc import abstractmethod


class NgramsExtractor:
    _cached_ngrams_per_config = {}
    _inner_ls = {}

    def __init__(self, config):
        self._config = config
        config_key = str(config)
        if config_key not in NgramsExtractor._cached_ngrams_per_config:
            NgramsExtractor._cached_ngrams_per_config[config_key] = {}
            
        self._parent_prefix = "p:" if self._config['add_prefix'] else ""
        self._child_prefix = "c:" if self._config['add_prefix'] else ""
        self._siblings_prefix = "s:" if self._config['add_prefix'] else ""

    def get_ngrams_from_target(self, target_tokens, freqs=False):
        config_key = str(self._config)
        cache_key = str(target_tokens)
        cached_ngrams = NgramsExtractor._cached_ngrams_per_config.get(config_key).get(cache_key)
        if cached_ngrams:
            return cached_ngrams
        else:
            NgramsExtractor._cached_ngrams_per_config[config_key][cache_key] = self._get_ngrams_from_target(target_tokens, freqs=freqs)
            for ls in NgramsExtractor._cached_ngrams_per_config[config_key][cache_key]:
                self.set_inner_ls(ls)

        return NgramsExtractor._cached_ngrams_per_config[config_key][cache_key]
    
    @classmethod
    def set_inner_ls(cls, ls):
        assert type(ls) == tuple
        ls_key = str(ls) if len(ls) > 1 else ls[0]        
        if ls_key not in cls._inner_ls:
            inner_ls_list = []
            for i in range(len(ls) + 1):
                for j in range(i + 1, len(ls) + 1):
                    inner_ls = ls[i:j]
                    if len(inner_ls) > 1:
                        inner_str = str(inner_ls)
                    else:
                        inner_str = str(inner_ls[0]) # handle atoms
                    inner_ls_list.append(inner_str)
            
            cls._inner_ls[ls_key] = inner_ls_list
        
    
    @classmethod
    def get_inner_ls(cls, ls):
        ls_key = str(ls)
        if ls_key not in cls._inner_ls:
            raise RuntimeError("ls not in _inner_ls dict")
        return cls._inner_ls[ls_key]
    
    
    @abstractmethod
    def _get_ngrams_from_target(self, target, freqs=False):
        pass
