# Maithili-Hindi Translation Models

## IndicTrans2 Model

### 1. Create Conda Environment
```bash
mkdir MaitH 1.0
cd MaitH 1.0
conda env create -f  IndicTrans2_environment.yml
conda activate IndicTrans2
mkdir it2
cd it2

```

### 2. Create Experiment Folder
```bash
mkdir indic-indic-exp
cd indic-indic-exp
mkdir train devtest vocab final_bin
```

#### Folder Structure
```
indic-indic-exp
├── train
│   ├── mai_Deva-hin_Deva
│       ├── train.mai_Deva
│       └── train.hin_Deva
│
├── devtest
│   ├── all
│   ├── mai_Deva-hin_Deva
│       ├── dev.mai_Deva
│       ├── dev.hin_Deva
│       ├── test.mai_Deva
│       └── test.hin_Deva
│
├── vocab
│   ├── model.SRC
│   ├── model.TGT
│   ├── vocab.SRC
│   └── vocab.TGT
└── final_bin
    ├── dict.SRC.txt
    └── dict.TGT.txt
```

### 3. Clone IndicTrans2 Repository 
```bash
git clone https://github.com/AI4Bharat/IndicTrans2
```

### 4. Install Dependencies
```bash

source install.sh
```


### 5. Data Preparation & Binarization
```bash
cd IndicTrans2
bash prepare_data_joint_finetuning.sh ../indic-indic-exp
```

### 6. Finetuning
download the model
https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
```bash

bash finetune.sh ../indic-indic-exp transformer_s /home/MaitH 1.0/it2/indic-indic-exp/model/checkpoint_best.pt
```

### 7. Inference
```bash
bash joint_translate.sh <infname> <outfname> <src_lang> <tgt_lang> <ckpt_dir>
bash joint_translate.sh ../indic-indic-exp/devtest/all/mai_Deva-hin_Deva/test.mai_Deva mai_hin_output.txt mai_Deva hin_Deva ../indic-indic-exp/
```

### 8. Evaluation: Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
cd ..
cd output
python BLEU4_chrF_TER_new.py
python COMET_METEOR.py
python BERTscore.py
```



## mT5 Model

### 1. Create Conda Environment
```bash
mkdir MaitH 1.0
cd MaitH 1.0
conda env create -f  mT5_environment.yml
conda activate mT5
mkdir mT5
cd mT5
```

### 2. Install Requirements
```bash
pip install -r requirement.txt
```

### 3. Training
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path google/mt5-base \
    --do_train --do_eval \
    --source_lang ma --target_lang hi --source_prefix "<2hi> " \
    --train_file /home/MaitH 1.0/mT5/data/train/train_mai_hin.json \
    --validation_file /home/MaitH 1.0/mT5/data/test/test_mai_hin.json \
    --test_file /home/MaitH 1.0/mT5/data/test/test_mai_hin.json \
    --output_dir checkpoints/mT5_mahi/ \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --num_train_epochs 7 \
    --predict_with_generate --save_strategy no \
    --metric_for_best_model bleu --overwrite_output_dir
```

### 4. Inference & BLEU Score Calculation
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path checkpoints/mT5_mahi \
    --do_predict \
    --source_lang ma --target_lang hi --source_prefix "<2hi> " \
    --validation_file /home/MaitH 1.0/mT5/data/test/test_mai_hin.json \
    --test_file /home/MaitH 1.0/mT5/data/test/test_mai_hin.json \
    --output_dir checkpoints/mT5_mahi/output \
    --per_device_eval_batch_size=4 \
    --predict_with_generate --overwrite_output_dir
```

### 5. ### 5. Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
cd /home/MaitH 1.0/mT5/output
python BLEU4_chrF_TER_new.py
python COMET_METEOR.py
python BERTscore.py
```

---

## mBART50 Model

### 1. Create Environment and Project Structure
```bash
mkdir MaitH 1.0
cd MaitH 1.0
conda env create -f  mBART50_environment.yml
conda activate mBART50
mkdir mBART50
cd mBART50
```

### 2. Download mBART50 Model from HuggingFace
Ensure that all dataset files are in `.json` format (e.g., `train.json`, `valid.json`, `test.json`).

### 3. Training
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --do_train --do_eval \
    --source_lang ma_XX --target_lang hi_IN \
    --train_file /home/MaitH 1.0/mBART50/data/train/train.json \
    --validation_file /home/MaitH 1.0/mBART50/data/dev/dev.json \
    --test_file /home/MaitH 1.0/mBART50/data/test/test.json \
    --output_dir checkpoint/mBART50_mai_hin/ \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs 7 \
    --predict_with_generate --save_strategy no \
    --metric_for_best_model bleu --overwrite_output_dir \
    --logging_dir ./mBART50_mai_hin/log --report_to tensorboard
```

### 4. Inference
```bash
python3 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path checkpoint/mBART50_mai_hin/ \
    --do_predict \
    --source_lang ma_XX --target_lang hi_IN \
    --validation_file /home/MaitH 1.0/mBART50/data/dev/dev.json \
    --test_file /home/MaitH 1.0/mBART50/data/test/test.json \
    --output_dir /home/MaitH 1.0/mBART50/output \
    --per_device_eval_batch_size=4 \
    --predict_with_generate --overwrite_output_dir
```

### 5. Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores

```bash
cd /home/MaitH 1.0/mBART50/output
python BLEU4_chrF_TER_new.py
python COMET_METEOR.py
python BERTscore.py
```

## NLLB-200 Model

### 1. Create Environment and Project Structure
```bash
mkdir MaitH 1.0
cd MaitH 1.0
conda env create -f  NLLB-200_environment.yml
conda activate NLLB-200
mkdir NLLB-200
cd NLLB-200
```

### 2. Download nllb-200-distilled-600M Model from HuggingFace, Ensure that all dataset files are in huggingface dataset format (e.g., `train`, `valid`, `test`).

```bash
cd /home/MaitH1.0/NLLB-200/scripts
python download.py
python convert_dataset.py
```
### 3. Training
```bash
python train.py
```
### 4. Inference

```bash
python inference.py
```
### 5. Compute BLEU4, chrF2, TER, COMET, METEOR, BERTScore Scores
```bash
python BLEU4_chrF_TER_new.py
python COMET_METEOR.py
python BERTscore.py
```

