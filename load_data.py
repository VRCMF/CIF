from datasets import load_dataset

# for lang in ['en', 'zh', 'de', 'es', 'fr', 'it', 'jap', 'nl', 'pl', 'pt', 'ru', 'ar', 'vi', 'hi', 'sw', 'ur']:
#     ds = load_dataset("INK-USC/xcsr", f"X-CSQA-{lang}", split="validation")
#     output_file = f"data/xcsr/xcsr_{lang}.jsonl"
#     print(f"Saving xcsr_{lang} → {output_file} ...")

#     ds.to_json(output_file,
#                orient="records",    # 每行一条 record
#                lines=True,
#                force_ascii=False)   # 保留 Unicode
#     # print(ds)


# for lang in ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']:
#     ds = load_dataset("facebook/xnli", lang)
#     print(ds)

for lang in ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'sw', 'bn', 'te', 'th']:


    ds = load_dataset("juletxara/mgsm", lang, split="test")

    ds.to_json(
        f"data/mgsm/mgsm_{lang}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
#     ds = load_dataset("Muennighoff/xwinograd", lang)
#     print(ds)


# for lang in ['en-zh', 'en-de', 'en-uk', 'en-ja', 'en-ru']:


#     ds = load_dataset("haoranxu/WMT23-Test", lang, split="test")

#     ds.to_json(
#         f"data/wmt/wmt_{lang}.jsonl",
#         orient="records",
#         lines=True,
#         force_ascii=False,


# for lang in ['en', 'ar', 'el', 'es', 'fr', 'hi', 'ru', 'tr', 'vi', 'zh']:
    
#     name_dict = {
#     "en": "eng",  # English
#     "ar": "ara",  # Arabic
#     "el": "ell",  # Greek
#     "es": "spa",  # Spanish
#     "fr": "fra",  # French
#     "hi": "hin",  # Hindi
#     "ru": "rus",  # Russian
#     "tr": "tur",  # Turkish
#     "vi": "vie",  # Vietnamese
#     "zh": "zho_simpl",  # Chinese
#     }
#     ds = load_dataset("gsarti/flores_101", name_dict[lang], split="dev")
#     ds.to_json(
#         f"data/flores/flores_{lang}.jsonl",
#         orient="records",
#         lines=True,
#         force_ascii=False,
#     )