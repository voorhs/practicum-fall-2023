if __name__ == "__main__":
    # inserter = Inserter(
    #     fraction=0.5,
    #     score_threshold=0.005,
    #     k=5,
    #     mask_utterance_level=True,
    #     fill_utterance_level=2,
    #     model='microsoft/mpnet-base',
    #     device='cuda'
    # )
    # inserter.from_file_system('inserter')
    
    # replacer = Replacer(
    #     k=3,
    #     fill_utterance_level=2,
    #     model='microsoft/mpnet-base',
    #     device='cuda'
    # )
    # replacer.from_file_system('replacer')

    # back_translator = BackTranslator(
    #     language='ru',
    #     device='cuda'
    # )
    # back_translator.from_file_system('back_translator')

    # model = 'meta-llama/Llama-2-13b-chat-hf'

    # tokenizer = AutoTokenizer.from_pretrained(model)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # llm = pipeline(
    #     "text-generation",
    #     model=AutoModelForCausalLM.from_pretrained(
    #         model,
    #         device_map='auto',
    #         load_in_4bit=True
    #     ),
    #     tokenizer=tokenizer
    # )

    # LlamaMaskFiller(llm, tokenizer, 'replace', fraction=0.2).from_file_system('llm_replacer')
    # LlamaMaskFiller(llm, tokenizer, 'insert', fraction=0.2).from_file_system('llm_inserter')
    # LlamaMaskFiller(llm, tokenizer, 'head', fraction=0.2).from_file_system('llm_head')
    # LlamaMaskFiller(llm, tokenizer, 'tail', fraction=0.2).from_file_system('llm_tail')

    # LlamaSummarizer(-5, llm, tokenizer).from_file_system('llm_summarizer')
    # LlamaVerbose(+5, llm, tokenizer).from_file_system('llm_verbose')
    # LlamaParaphraser('formal', llm, tokenizer).from_file_system('llm_formal')
    # LlamaParaphraser('informal', llm, tokenizer).from_file_system('llm_informal')
    # LlamaParaphraser('technical', llm, tokenizer).from_file_system('llm_technical')
    # LlamaParaphraser('persuasive', llm, tokenizer).from_file_system('llm_persuasive')
    # LlamaParaphraser('creative', llm, tokenizer).from_file_system('llm_creative')
    # LlamaParaphraser('playful', llm, tokenizer).from_file_system('llm_playful')
    
    # listwise_shuffler = ListwiseShuffler(
    #     ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/listwise-utterance-transformer-amazon-resumed/checkpoints/last.ckpt',
    #     device='cuda:0'
    # )
    # listwise_shuffler.from_file_system('listwise_shuffler')

    import os
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', dest='method', required=True, choices=[
        'back-translate', 'insert', 'replace',
        'listwise-shuffler', 'prune', 'shuffle'
    ])
    ap.add_argument('--seed', dest='seed', default=0)
    ap.add_argument('--name', dest='name', required=True)
    ap.add_argument('--cuda', dest='cuda', required=True)
    ap.add_argument('--path-in', dest='path_in', default=os.path.join(os.getcwd(), 'src/mylib/data/train/source/train'))
    args = ap.parse_args()

    from mylib.utils.training import init_environment
    init_environment(args)

    json_chunks = sorted([filename for filename in os.listdir(args.path_in) if filename.endswith('.json')])
    args.path_out = os.path.join(os.getcwd(), 'src/mylib/data/augmented', args.name)

    print('result is going to be saved to', args.path_out)

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)
    
    from mylib.augmentations import *

    if args.method == 'back-translate':
        augmenter = BackTranslator(language='ru', device='cuda')
    elif args.method == 'insert':
        augmenter = Inserter(
            fraction=0.5,
            score_threshold=0.005,
            k=5,
            mask_utterance_level=True,
            fill_utterance_level=2,
            model='microsoft/mpnet-base',
            device='cuda'
        )
    elif args.method == 'replace':
        augmenter = Replacer(
            k=3,
            fill_utterance_level=2,
            model='microsoft/mpnet-base',
            device='cuda'
        )
    # elif args.method == 'listwise-shuffler':
    #     augmenter = ListwiseShuffler(device='cuda')
    elif args.method == 'prune':
        augmenter = Pruner(device='cuda')
    elif args.method == 'shuffle':
        augmenter = Shuffler(device='cuda')
        exit()
    
    import json
    from tqdm import tqdm
    
    for chunk in tqdm(json_chunks[80:]):
        dialogues = json.load(open(os.path.join(args.path_in, chunk), 'r'))
        clean_dialogues = [dia for dia in dialogues if dia is not None]
        augmented = augmenter(clean_dialogues)
        if args.method == 'back-translate':
            augmented, ru = augmented
            json.dump(ru, open(os.path.join(args.path_out, 'ru-' + chunk), 'w'))
        for i, dia in enumerate(dialogues):
            if dia is None:
                augmented.insert(i, None)
        json.dump(augmented, open(os.path.join(args.path_out, chunk), 'w'))
