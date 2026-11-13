#!/usr/bin/env python3
"""
Preprocessing script to convert OPUS data from split/ directory 
to ncwm format for INCLINE_2 training.

This script:
1. Reads parallel documents from data/split/{lang}/ directories
2. Extracts sentences from documents
3. Aligns sentences at paragraph level
4. Writes train.en and train.{lang} files in ncwm format
"""

import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Set

# Languages to process (matching intervention.py)
# Note: Only languages that exist in split/ directory will be processed
# seen_langs = ['ar', 'es', 'fr', 'hi', 'id', 'pt', 'sw', 'ta', 'vi', 'zh']
# unseen_langs = ['de', 'el', 'et', 'it', 'ja', 'nl', 'ru', 'th', 'tr', 'uk']
# LANGS = seen_langs + unseen_langs
# unseen_langs = ['el', 'hi', 'ja']
# unseen_langs = ['sw', 'el', 'et', 'th', 'tr', 'uk', 'ta', 'vi']
LANGS = ['ur']

# Base directories
SPLIT_DIR = Path("./data/split")
BACKUP_DIR = Path("./data/backup")
OUTPUT_BASE_DIR = Path("./data/ncwm")
MAX_OUTPUT_LINES = 500  # Limit output files to 1000 lines
MIN_SENTENCE_LENGTH = 30  # Minimum character length for sentences (increased from 10)


def get_available_languages() -> Set[str]:
    """Get list of available languages from split directory."""
    if not SPLIT_DIR.exists():
        return set()
    return {d.name for d in SPLIT_DIR.iterdir() if d.is_dir()}


def is_valid_english_sentence(text: str) -> bool:
    """
    Check if English sentence contains only English letters and punctuation.
    
    Args:
        text: Text to check
    
    Returns:
        True if text contains only English letters and punctuation, False otherwise
    """
    import string
    # Allowed characters: English letters (a-z, A-Z), spaces, and punctuation
    allowed_chars = set(string.ascii_letters + string.punctuation + string.whitespace)
    text_chars = set(text)
    
    # Check if all characters in text are in allowed set
    return text_chars.issubset(allowed_chars)


def is_likely_english(text: str) -> bool:
    """
    Check if text is likely English (not other Latin-script languages).
    Uses common English words and patterns to distinguish from other languages.
    
    Args:
        text: Text to check
    
    Returns:
        True if text appears to be English, False otherwise
    """
    import string
    import re
    
    # If text doesn't contain only ASCII letters, it's not English
    if not is_valid_english_sentence(text):
        return False
    
    # Common English words (articles, prepositions, common verbs)
    english_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'it', 'its', 'they', 'them', 'their', 'there', 'here',
        'where', 'when', 'what', 'who', 'which', 'how', 'why', 'not', 'no',
        'yes', 'if', 'then', 'else', 'than', 'more', 'most', 'less', 'very',
        'much', 'many', 'some', 'any', 'all', 'each', 'every', 'other',
        'another', 'one', 'two', 'first', 'second', 'last', 'new', 'old',
        'good', 'bad', 'great', 'small', 'large', 'big', 'little', 'high',
        'low', 'long', 'short', 'time', 'year', 'day', 'week', 'month',
        'way', 'man', 'thing', 'world', 'life', 'hand', 'part', 'child',
        'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
        'company', 'number', 'group', 'problem', 'fact', 'system', 'program',
        'question', 'work', 'government', 'public', 'national', 'international'
    }
    
    # Convert to lowercase and extract words
    text_lower = text.lower()
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-z]+\b', text_lower)
    
    if len(words) == 0:
        return False
    
    # Count how many words are common English words
    english_word_count = sum(1 for word in words if word in english_words)
    english_ratio = english_word_count / len(words)
    
    # If more than 30% of words are common English words, likely English
    return english_ratio > 0.3


def is_not_english(text: str, source_lang: str) -> bool:
    """
    Check if text is NOT English (i.e., it's in the source language).
    For languages using Latin script (it, sw, etc.), uses heuristics to distinguish.
    For non-Latin scripts (zh, ar, etc.), checks if text contains non-ASCII characters.
    
    Args:
        text: Text to check
        source_lang: Source language code
    
    Returns:
        True if text is likely NOT English, False if it might be English
    """
    import string
    
    # Non-Latin script languages: check for non-ASCII characters
    non_latin_langs = ['zh', 'ar', 'hi', 'th', 'ta', 'ja', 'ko', 'ru', 'uk', 'el', 'he', 'kk']
    if source_lang in non_latin_langs:
        # For non-Latin scripts, if text contains non-ASCII, it's likely the source language
        text_chars = set(text)
        ascii_chars = set(string.ascii_letters + string.punctuation + string.digits + string.whitespace)
        non_ascii_chars = text_chars - ascii_chars
        # If has significant non-ASCII characters, it's likely the source language
        if len(non_ascii_chars) > 0:
            return True
        # If only ASCII, check if it looks like English
        # For non-Latin script languages, pure ASCII text is likely English, not the source language
        if is_likely_english(text):
            return False
        # If not clearly English, be conservative and reject (might be English)
        return False
    
    # Latin script languages: use heuristics to distinguish from English
    # For these languages, we check if it's likely English, and reject if so
    latin_langs = ['it', 'sw', 'es', 'fr', 'de', 'pt', 'nl', 'id', 'vi', 'tr', 'et']
    if source_lang in latin_langs:
        # If text looks like English, reject it (it's NOT the source language)
        if is_likely_english(text):
            return False
        # Otherwise, assume it's the source language
        return True
    
    # Default: if we can't determine, be conservative and reject
    return False


def extract_sentences_from_doc(file_path: Path) -> List[str]:
    """
    Extract sentences from a document file.
    Documents have <HEADLINE> and <P> tags.
    Returns a list of sentences.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sentences = []
    # Remove HTML-like tags
    content = re.sub(r'<HEADLINE>', '', content)
    content = re.sub(r'</HEADLINE>', '', content)
    content = re.sub(r'<P>', '', content)
    content = re.sub(r'</P>', '', content)
    
    # Split by paragraphs first (double newlines) or single newlines
    paragraphs = re.split(r'\n\s*\n', content)
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Split paragraph into sentences using multiple delimiters
        # Split by sentence-ending punctuation followed by space or newline
        # Pattern: period, question mark, or exclamation mark followed by space/newline
        # Split on: . ! ? followed by whitespace (space, tab, newline) or end of string
        para_sentences = re.split(r'(?<=[.!?])(?:\s+|$)', para)
        
        for sent in para_sentences:
            sent = sent.strip()
            # Filter very short sentences and ensure minimum length
            if sent and len(sent) >= MIN_SENTENCE_LENGTH:
                # Ensure sentence ends with punctuation
                if not sent.endswith('.') and not sent.endswith('?') and not sent.endswith('!'):
                    sent += '.'
                sentences.append(sent)
    
    return sentences


def extract_paragraphs_from_doc(file_path: Path) -> List[str]:
    """
    Extract paragraphs from a document file.
    This provides better alignment than sentence-level extraction.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    paragraphs = []
    current_para = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and tags
        if not line or line.startswith('<HEADLINE>') or line.startswith('</HEADLINE>'):
            continue
        
        # Start new paragraph on <P> tag
        if line == '<P>':
            if current_para:
                para_text = ' '.join(current_para).strip()
                if para_text and len(para_text) > 20:  # Filter very short paragraphs
                    paragraphs.append(para_text)
            current_para = []
        elif line and not line.startswith('<'):
            # Add content line to current paragraph
            current_para.append(line)
    
    # Don't forget the last paragraph
    if current_para:
        para_text = ' '.join(current_para).strip()
        if para_text and len(para_text) > 20:
            paragraphs.append(para_text)
    
    return paragraphs


def find_common_files(lang1: str, lang2: str) -> List[str]:
    """
    Find common files between two language directories.
    Returns list of filenames that exist in both directories.
    """
    dir1 = SPLIT_DIR / lang1
    dir2 = SPLIT_DIR / lang2
    
    if not dir1.exists() or not dir2.exists():
        return []
    
    files1 = set(f.name for f in dir1.glob("*.txt"))
    files2 = set(f.name for f in dir2.glob("*.txt"))
    
    common_files = sorted(files1 & files2)
    return common_files


def process_tsv_file(tsv_file: Path, max_lines: int = MAX_OUTPUT_LINES, min_length: int = MIN_SENTENCE_LENGTH, seed: int = 42):
    """
    Process a TSV file from backup directory.
    TSV format: en_sentence\tlang_sentence\tscore
    
    Args:
        tsv_file: Path to TSV file
        max_lines: Maximum number of lines to output
        min_length: Minimum character length for sentences
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (target_sentences, source_sentences) lists
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # First pass: collect all valid sentence pairs
    all_target_sentences = []
    all_source_sentences = []
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # import ipdb
            # ipdb.set_trace()
            parts = line.split('\t')
            if len(parts) >= 2:
                en_sent = parts[0].strip()
                lang_sent = parts[1].strip()
                
                # Filter out very short sentences
                if len(en_sent) >= min_length and len(lang_sent) >= min_length:
                    # Filter out English sentences that contain non-English characters
                    if 'el' in str(tsv_file):
                    # if is_valid_english_sentence(en_sent):
                        all_target_sentences.append(en_sent)
                        all_source_sentences.append(lang_sent)
                    else:
                        if is_valid_english_sentence(en_sent):
                            all_target_sentences.append(en_sent)
                            all_source_sentences.append(lang_sent)
    
    # Randomly sample up to max_lines
    if len(all_target_sentences) <= max_lines:
        return all_target_sentences, all_source_sentences
    else:
        # Create indices and randomly sample
        indices = list(range(len(all_target_sentences)))
        random.shuffle(indices)
        selected_indices = indices[:max_lines]  # Keep random order
        
        target_sentences = [all_target_sentences[i] for i in selected_indices]
        source_sentences = [all_source_sentences[i] for i in selected_indices]
        
        return target_sentences, source_sentences


def process_language_pair(source_lang: str, target_lang: str = 'en', use_backup: bool = False):
    """
    Process a language pair and generate ncwm format files.
    
    Args:
        source_lang: Source language code (e.g., 'zh', 'de')
        target_lang: Target language code (default: 'en')
        use_backup: If True, use TSV files from backup directory
    """
    print(f"Processing {target_lang}-{source_lang}...")
    
    # Create output directory
    output_dir = OUTPUT_BASE_DIR / f"{target_lang}-{source_lang}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_sentences = []
    source_sentences = []
    
    if use_backup:
        # Process TSV file from backup directory
        tsv_file = BACKUP_DIR / f"{target_lang}_{source_lang}.tsv"
        if not tsv_file.exists():
            print(f"  Warning: TSV file not found: {tsv_file}")
            return
        
        print(f"  Reading from {tsv_file}")
        target_sentences, source_sentences = process_tsv_file(
            tsv_file, 
            max_lines=MAX_OUTPUT_LINES, 
            min_length=MIN_SENTENCE_LENGTH
        )
        print(f"  Extracted {len(target_sentences)} sentence pairs (limited to {MAX_OUTPUT_LINES}, min length: {MIN_SENTENCE_LENGTH})")
    
    else:
        # Process from split directory
        common_files = find_common_files(target_lang, source_lang)
        print(f"  Found {len(common_files)} common files")
        
        if len(common_files) == 0:
            print(f"  Warning: No common files found for {target_lang}-{source_lang}")
            return
        
        # Randomly sample files first (up to 1000 files)
        MAX_FILES = 1000
        total_files = len(common_files)
        if total_files > MAX_FILES:
            random.seed(42)  # For reproducibility
            common_files = random.sample(common_files, MAX_FILES)
            print(f"  Randomly selected {len(common_files)} files from {total_files} total files")
        else:
            print(f"  Processing all {len(common_files)} files")
        
        # First pass: collect all valid sentence pairs
        all_target_sentences = []
        all_source_sentences = []
        
        processed_files = 0
        for filename in common_files:
            target_file = SPLIT_DIR / target_lang / filename
            source_file = SPLIT_DIR / source_lang / filename
            
            try:
                # Extract sentences from documents
                target_sents = extract_sentences_from_doc(target_file)
                source_sents = extract_sentences_from_doc(source_file)
                
                # Align sentences: use minimum length to avoid misalignment
                min_len = min(len(target_sents), len(source_sents))
                
                for i in range(min_len):
                    target_sent = target_sents[i]
                    source_sent = source_sents[i]
                    
                    # Filter out very short sentences
                    if len(target_sent) >= MIN_SENTENCE_LENGTH and len(source_sent) >= MIN_SENTENCE_LENGTH:
                        # Verify target (English) sentence is valid English
                        if not is_valid_english_sentence(target_sent):
                            continue
                        
                        # Verify source sentence is NOT English (i.e., it's in the source language)
                        if not is_not_english(source_sent, source_lang):
                            # Source sentence looks like English, skip this pair
                            continue
                        
                        # Both sentences pass validation, add them
                        all_target_sentences.append(target_sent)
                        all_source_sentences.append(source_sent)
                
                processed_files += 1
                
            except Exception as e:
                print(f"  Warning: Error processing {filename}: {e}")
                continue
        
        # Randomly sample up to MAX_OUTPUT_LINES
        if len(all_target_sentences) <= MAX_OUTPUT_LINES:
            target_sentences = all_target_sentences
            source_sentences = all_source_sentences
        else:
            # Create indices and randomly sample
            indices = list(range(len(all_target_sentences)))
            random.seed(42)  # For reproducibility
            random.shuffle(indices)
            selected_indices = indices[:MAX_OUTPUT_LINES]  # Keep random order
            
            target_sentences = [all_target_sentences[i] for i in selected_indices]
            source_sentences = [all_source_sentences[i] for i in selected_indices]
        
        print(f"  Processed {processed_files} files")
        print(f"  Extracted {len(target_sentences)} sentence pairs (limited to {MAX_OUTPUT_LINES}, min length: {MIN_SENTENCE_LENGTH})")
    
    # Limit to MAX_OUTPUT_LINES
    target_sentences = target_sentences[:MAX_OUTPUT_LINES]
    source_sentences = source_sentences[:MAX_OUTPUT_LINES]
    
    # Write output files
    target_output_file = output_dir / f"train.{target_lang}"
    source_output_file = output_dir / f"train.{source_lang}"
    
    with open(target_output_file, 'w', encoding='utf-8') as f:
        for sent in target_sentences:
            f.write(sent + '\n')
    
    with open(source_output_file, 'w', encoding='utf-8') as f:
        for sent in source_sentences:
            f.write(sent + '\n')
    
    print(f"  Written {len(target_sentences)} lines to {target_output_file}")
    print(f"  Written {len(source_sentences)} lines to {source_output_file}")
    print()


def main():
    """Main function to process all language pairs."""
    print("=" * 60)
    print("OPUS to ncwm Preprocessing")
    print(f"Max output lines per file: {MAX_OUTPUT_LINES}")
    print("=" * 60)
    print()
    
    # Get available languages from split directory
    available_langs = get_available_languages()
    print(f"Available languages in split/: {sorted(available_langs)}")
    
    # Get available TSV files from backup directory
    backup_langs = set()
    if BACKUP_DIR.exists():
        for tsv_file in BACKUP_DIR.glob("en_*.tsv"):
            lang = tsv_file.stem.split('_')[1]  # Extract lang from en_{lang}.tsv
            backup_langs.add(lang)
        print(f"Available languages in backup/: {sorted(backup_langs)}")
    print()
    
    processed_count = 0
    
    # Process each language pair (en-{lang})
    for lang in LANGS:
        if lang == 'en':
            continue  # Skip en-en
        
        try:
            # Check if TSV file exists in backup
            tsv_file = BACKUP_DIR / f"en_{lang}.tsv"
            use_backup = tsv_file.exists()
            
            if use_backup:
                print(f"Using backup TSV file for {lang}")
                process_language_pair(lang, 'en', use_backup=True)
                processed_count += 1
            elif lang in available_langs:
                print(f"Using split directory for {lang}")
                process_language_pair(lang, 'en', use_backup=False)
                processed_count += 1
            else:
                print(f"Skipping {lang} (not found in split directory or backup)")
                continue
                
        except Exception as e:
            print(f"Error processing en-{lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 60)
    print(f"Preprocessing complete! Processed {processed_count} language pairs.")
    print("=" * 60)


if __name__ == "__main__":
    main()

