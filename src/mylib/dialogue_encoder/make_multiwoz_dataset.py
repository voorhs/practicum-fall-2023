CHUNK_SIZE = 512
tokenizer = 'roberta-base'
path_out = '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/multiwoz_truncated'

if __name__ == "__main__":
    from datasets import load_dataset
    import json
    import os
    from filter_dataset_by_length import filter_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from filter_dataset_by_length import is_short_enough

    dataset = load_dataset('multi_woz_v22')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    def save_chunk(dataset_slice, path_out, i_chunk):
        """Saves given slice as two json files: dia-{i_chunk}.json and services-{i_chunk}.json. Skips dias with exceeding length (see `is_short_enough()` in `filter_dataset_by_length.py`)"""
        chunk_dias = []
        chunk_services = []
        for item in dataset_slice:
            uts = item['turns']['utterance']
            speakers = item['turns']['speaker']
            dia = [{'utterance': ut, 'speaker': sp} for ut, sp in zip(uts, speakers)]

            if not is_short_enough(dia, tokenizer):
                print('rejected dia')
                continue
            
            chunk_dias.append(dia)
            services = item['services']
            chunk_services.append(services)
        
        dia_path = os.path.join(path_out, f'dia-{i_chunk}.json')
        services_path = os.path.join(path_out, f'services-{i_chunk}.json')
        json.dump(chunk_dias, open(dia_path, 'w'))
        json.dump(chunk_services, open(services_path, 'w'))

    for split in ['train', 'validation', 'test']:
        data_split = dataset[split]
        
        cur_path = os.path.join(path_out, split)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        
        split_size = len(data_split)
        for i_chunk, start in tqdm(enumerate(range(0, split_size, CHUNK_SIZE)), desc=f'making chunks for {split}'):
            end = start + CHUNK_SIZE
            dataset_slice = [data_split[i] for i in range(start, min(end, split_size))]
            
            save_chunk(dataset_slice, cur_path, i_chunk)
    
    
