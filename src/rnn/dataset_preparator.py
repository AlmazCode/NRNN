import re
import os
from pathlib import Path
from typing import Optional, List


class DatasetPreparator:
    """Advanced text dataset preparation for training language models.
    
    Features:
        - Multi-file loading with encoding detection
        - Configurable cleaning levels
        - Metadata removal
        - Character filtering
        - Comprehensive statistics
    """
    
    def __init__(self):
        self.text = ""
        self.stats = {}
        
    def load_file(self, filepath: str, encodings: List[str] = None) -> str:
        """Load a single file with automatic encoding detection.
        
        Args:
            filepath: Path to the file
            encodings: List of encodings to try (default: utf-8, windows-1251, cp1252)
            
        Returns:
            File content as string
        """
        if encodings is None:
            encodings = ['utf-8', 'windows-1251', 'cp1252', 'latin-1']
        
        filepath = Path(filepath)
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"✓ Loaded {filepath.name}: {len(content):,} chars ({encoding})")
                    return content
            except (UnicodeDecodeError, LookupError):
                continue
        
        print(f"✗ Failed to load {filepath}")
        return ""
    
    def load_multiple_files(self, folder_path: str, pattern: str = '*.txt') -> str:
        """Load all matching files from a folder.
        
        Args:
            folder_path: Path to folder
            pattern: Glob pattern for files (default: *.txt)
            
        Returns:
            Combined text from all files
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"✗ Folder not found: {folder_path}")
            return ""
        
        all_text = []
        files = sorted(folder.glob(pattern))
        
        if not files:
            print(f"✗ No files matching '{pattern}' in {folder_path}")
            return ""
        
        for filepath in files:
            content = self.load_file(filepath)
            if content:
                all_text.append(content)
        
        combined = '\n\n'.join(all_text)
        print(f"\n✓ Total files loaded: {len(all_text)}")
        print(f"✓ Combined size: {len(combined):,} chars ({len(combined)/(1024*1024):.2f} MB)")
        return combined
    
    def merge_short_lines(self, text: str, max_line_length: int = 70, merge_short_paragraphs: bool = True) -> str:
        """Merge short lines that are parts of the same paragraph.
        
        This fixes books where each sentence is on a separate line.
        
        Args:
            text: Input text
            max_line_length: Lines shorter than this are merged (unless dialogue)
            merge_short_paragraphs: If True, merge consecutive short paragraphs (< 50 chars each)
            
        Returns:
            Text with merged lines
        """
        lines = text.split('\n')
        result = []
        buffer = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip completely empty lines
            if not stripped:
                # Flush buffer
                if buffer:
                    result.append(' '.join(buffer))
                    buffer = []
                result.append('')
                continue
            
            # Detect explicit dialogue markers (keep these separate always)
            starts_with_dash = stripped.startswith(('—', '–', '- '))
            starts_with_quote = stripped.startswith(('"', '«'))
            
            # Chapter/section headers
            is_header = (
                (len(stripped) < 60 and stripped.isupper()) or
                (len(stripped) < 50 and stripped.istitle() and '.' not in stripped and ',' not in stripped)
            )
            
            if is_header:
                if buffer:
                    result.append(' '.join(buffer))
                    buffer = []
                result.append(stripped)
                result.append('')
                continue
            
            # Handle explicit dialogue with dashes (keep separate)
            if starts_with_dash:
                if buffer:
                    result.append(' '.join(buffer))
                    buffer = []
                result.append(stripped)
                # Don't add extra empty line - let natural paragraph breaks handle it
                continue
            
            # For everything else (narrative, dialogue without dashes), accumulate
            buffer.append(stripped)
            
            # Decide when to flush buffer
            # Flush if line ends with strong punctuation AND meets conditions
            if stripped and stripped[-1] in '.!?…':
                should_flush = False
                
                # Always flush if line is long
                if len(stripped) > max_line_length:
                    should_flush = True
                # Flush if at end of text
                elif i + 1 >= len(lines):
                    should_flush = True
                # Look ahead
                elif i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    # Flush if next is:
                    # - empty line (paragraph break)
                    # - starts with capital letter (new sentence)
                    # - is dialogue with dash
                    # - is a header
                    if (not next_stripped or
                        (next_stripped and next_stripped[0].isupper() and not stripped.endswith(('.', '...'))) or
                        next_stripped.startswith(('—', '–', '- ')) or
                        (len(next_stripped) < 60 and next_stripped.isupper())):
                        should_flush = True
                
                if should_flush:
                    result.append(' '.join(buffer))
                    buffer = []
        
        # Flush remaining buffer
        if buffer:
            result.append(' '.join(buffer))
        
        # Post-processing: merge consecutive short paragraphs (dialogue fragments)
        if merge_short_paragraphs:
            final_result = []
            i = 0
            while i < len(result):
                line = result[i]
                
                # If empty line, keep it
                if not line.strip():
                    final_result.append(line)
                    i += 1
                    continue
                
                # If line starts with dash, keep as-is
                if line.startswith(('—', '–', '- ')):
                    final_result.append(line)
                    i += 1
                    continue
                
                # Accumulate short paragraphs
                short_buffer = [line]
                j = i + 1
                
                # Look ahead for more short paragraphs separated by single empty lines
                while j < len(result):
                    if not result[j].strip():
                        # Empty line found
                        if j + 1 < len(result):
                            next_para = result[j + 1].strip()
                            # If next paragraph is short and not dialogue, merge it
                            if (next_para and 
                                len(next_para) < 50 and 
                                not next_para.startswith(('—', '–', '- ')) and
                                not next_para.isupper()):
                                short_buffer.append(next_para)
                                j += 2  # Skip empty line and the merged paragraph
                            else:
                                break
                        else:
                            break
                    else:
                        break
                
                # Add merged result
                if len(short_buffer) > 1:
                    final_result.append(' '.join(short_buffer))
                    final_result.append('')  # Add one empty line after merged block
                else:
                    final_result.append(short_buffer[0])
                
                i = j
            
            result = final_result
        
        # Final pass: remove excessive empty lines (3+ → 2)
        final_lines = []
        empty_count = 0
        
        for line in result:
            if not line.strip():
                empty_count += 1
                if empty_count <= 1:  # Keep max 1 empty line (which becomes \n\n)
                    final_lines.append(line)
            else:
                empty_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def clean_text(self, text: str, level: str = 'medium') -> str:
        """Clean text with different intensity levels.
        
        Args:
            text: Input text
            level: 'light', 'medium', 'aggressive'
            
        Returns:
            Cleaned text
        """
        # Remove BOM and normalize line endings
        text = text.replace('\ufeff', '')
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Convert common unicode variations to standard forms
        text = text.replace('—', '—')  # em dash
        text = text.replace('–', '-')  # en dash to hyphen
        text = text.replace('…', '...')  # ellipsis
        text = text.replace('«', '"').replace('»', '"')  # quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        if level == 'light':
            # Minimal cleaning - only fix obvious issues
            text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
            text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing spaces
            
        elif level == 'medium':
            # Standard cleaning
            text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
            text = re.sub(r'[ \t]{2,}', ' ', text)  # Multiple spaces to one
            text = re.sub(r'\t', ' ', text)  # Tabs to spaces
            text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing spaces
            # Fix spacing around punctuation
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1\2', text)
            
        elif level == 'aggressive':
            # Heavy cleaning
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]{2,}', ' ', text)
            text = re.sub(r'\t', ' ', text)
            text = re.sub(r'[ \t]+\n', '\n', text)
            
            # Remove URLs and emails
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'\S+@\S+\.\S+', '', text)
            
            # Remove lines that are mostly numbers (tables, etc.)
            # BUT keep lines with normal text that contains numbers
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    cleaned_lines.append(line)
                    continue
                
                # Count actual digits vs letters
                digits = sum(c.isdigit() for c in line)
                letters = sum(c.isalpha() for c in line)
                
                # Keep line if:
                # - Has more letters than digits
                # - Has at least some letters and digit ratio < 70%
                if letters > digits or (letters > 5 and digits / len(line) < 0.7):
                    cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
            
            # Remove very short lines (likely junk)
            lines = text.split('\n')
            cleaned_lines = []
            for i, line in enumerate(lines):
                # Keep empty lines for paragraph separation
                if not line.strip():
                    cleaned_lines.append(line)
                # Keep lines with reasonable length or that are part of dialogue
                elif len(line) > 3 or line.startswith('—'):
                    cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
        
        # Final cleanup: strip trailing spaces from each line
        # but preserve empty lines (paragraph separators)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Now clean up excessive empty lines
        # Replace 3+ consecutive newlines with exactly 2 (to preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_metadata(self, text: str, aggressive: bool = False) -> str:
        """Remove metadata from books (headers, footers, etc.).
        
        Args:
            text: Input text
            aggressive: If True, also remove common footer patterns throughout text
            
        Returns:
            Text with metadata removed
        """
        # Common metadata markers
        markers_start = [
            'текст предоставлен',
            'скачано с',
            'downloaded from',
            'project gutenberg',
            '*** start of',
            'источник:',
            'provided by',
            'courtesy of',
            'transcribed by',
        ]
        
        markers_end = [
            '*** end of',
            'конец ознакомительного фрагмента',
            'end of project gutenberg',
            'end of the project',
            'конец книги',
        ]
        
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find start of actual content (search first 100 lines)
        for i, line in enumerate(lines[:100]):
            if any(marker in line.lower() for marker in markers_start):
                start_idx = i + 1
        
        # Find end of actual content (search last 100 lines)
        for i in range(max(0, len(lines) - 100), len(lines)):
            line = lines[i]
            if any(marker in line.lower() for marker in markers_end):
                end_idx = i
                break
        
        text = '\n'.join(lines[start_idx:end_idx])
        
        # Remove common page numbers and footnotes
        if aggressive:
            text = re.sub(r'\n\d+\n', '\n', text)  # Standalone page numbers
            text = re.sub(r'\n\[\d+\].*?\n', '\n', text)  # Footnote markers
        
        return text
    
    def filter_by_charset(self, 
                         text: str, 
                         keep_cyrillic: bool = True,
                         keep_latin: bool = True,
                         keep_digits: bool = True,
                         keep_punctuation: bool = True,
                         custom_chars: str = '') -> str:
        """Filter text by allowed character set.
        
        Args:
            text: Input text
            keep_cyrillic: Keep Cyrillic (Russian) characters
            keep_latin: Keep Latin (English) characters
            keep_digits: Keep digits
            keep_punctuation: Keep punctuation marks
            custom_chars: Additional characters to keep
            
        Returns:
            Filtered text
        """
        allowed_chars = set()
        
        if keep_cyrillic:
            allowed_chars.update('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
            allowed_chars.update('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        
        if keep_latin:
            allowed_chars.update('abcdefghijklmnopqrstuvwxyz')
            allowed_chars.update('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        if keep_digits:
            allowed_chars.update('0123456789')
        
        if keep_punctuation:
            # Russian and English punctuation
            allowed_chars.update('.,!?;:—-–()[]{}«»""\'"…\n\t ')
        else:
            allowed_chars.update('\n ')
        
        # Add custom characters
        allowed_chars.update(custom_chars)
        
        # Filter
        filtered_text = ''.join(c for c in text if c in allowed_chars)
        
        # Clean up excessive whitespace that may have been created
        filtered_text = re.sub(r' {2,}', ' ', filtered_text)
        filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
        
        return filtered_text
    
    def get_statistics(self, text: str, show_chars: bool = True) -> dict:
        """Get comprehensive text statistics.
        
        Args:
            text: Input text
            show_chars: Whether to print unique characters
            
        Returns:
            Dictionary with statistics
        """
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        words = text.split()
        
        # Character distribution
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        stats = {
            'total_chars': len(text),
            'unique_chars': len(set(text)),
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'total_words': len(words),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
            'size_mb': len(text) / (1024 * 1024),
            'most_common_chars': sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        print("\n" + "="*60)
        print("DATASET STATISTICS".center(60))
        print("="*60)
        print(f"Total characters:     {stats['total_chars']:>15,}")
        print(f"Unique characters:    {stats['unique_chars']:>15}")
        print(f"Total lines:          {stats['total_lines']:>15,}")
        print(f"Non-empty lines:      {stats['non_empty_lines']:>15,}")
        print(f"Total words:          {stats['total_words']:>15,}")
        print(f"Avg word length:      {stats['avg_word_length']:>15.1f}")
        print(f"Avg line length:      {stats['avg_line_length']:>15.1f}")
        print(f"Size (MB):            {stats['size_mb']:>15.2f}")
        
        print(f"\nMost common characters:")
        for char, count in stats['most_common_chars']:
            char_display = repr(char) if char in '\n\t ' else char
            print(f"  {char_display:>5} : {count:>10,} ({count/len(text)*100:.1f}%)")
        
        if show_chars:
            # Group characters by type
            cyrillic = sorted([c for c in set(text) if 'а' <= c.lower() <= 'я' or c == 'ё'])
            latin = sorted([c for c in set(text) if 'a' <= c.lower() <= 'z'])
            digits = sorted([c for c in set(text) if c.isdigit()])
            punctuation = sorted([c for c in set(text) if c in '.,!?;:—-–()[]{}«»""\'"…'])
            other = sorted([c for c in set(text) if c not in cyrillic + latin + digits + punctuation and c not in '\n\t '])
            
            print(f"\nCharacter breakdown:")
            print(f"  Cyrillic ({len(cyrillic)}): {''.join(cyrillic)}")
            print(f"  Latin ({len(latin)}): {''.join(latin)}")
            print(f"  Digits ({len(digits)}): {''.join(digits)}")
            print(f"  Punctuation ({len(punctuation)}): {''.join(punctuation)}")
            if other:
                print(f"  Other ({len(other)}): {repr(''.join(other))}")
        
        print("="*60 + "\n")
        
        self.stats = stats
        return stats
    
    def validate_dataset(self, text: str, min_size_mb: float = 0.01) -> bool:
        """Validate that dataset meets minimum requirements.
        
        Args:
            text: Text to validate
            min_size_mb: Minimum size in MB
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not text.strip():
            print("✗ Validation failed: Empty text")
            return False
        
        size_mb = len(text) / (1024 * 1024)
        if size_mb < min_size_mb:
            print(f"✗ Validation failed: Size {size_mb:.2f} MB < minimum {min_size_mb} MB")
            return False
        
        unique_chars = len(set(text))
        if unique_chars < 10:
            print(f"✗ Validation failed: Only {unique_chars} unique characters")
            return False
        
        print(f"✓ Validation passed: {size_mb:.2f} MB, {unique_chars} unique chars")
        return True
    
    def prepare_dataset(self, 
                       input_path: str, 
                       output_path: str = 'prepared_dataset.txt',
                       clean_level: str = 'medium',
                       remove_meta: bool = True,
                       merge_lines: bool = True,
                       filter_charset: bool = True,
                       keep_latin: bool = True,
                       min_size_mb: float = 0.01) -> Optional[str]:
        """Complete dataset preparation pipeline.
        
        Args:
            input_path: Path to file or folder
            output_path: Where to save prepared dataset
            clean_level: 'light', 'medium', 'aggressive'
            remove_meta: Remove metadata from books
            merge_lines: Merge short lines into paragraphs (fix broken formatting)
            filter_charset: Filter characters to allowed set
            keep_latin: Keep Latin characters (in addition to Cyrillic)
            min_size_mb: Minimum output size for validation
            
        Returns:
            Prepared text or None if failed
        """
        print("="*60)
        print("DATASET PREPARATION PIPELINE".center(60))
        print("="*60 + "\n")
        
        # Step 1: Load
        print("[1/6] Loading text...")
        if os.path.isdir(input_path):
            text = self.load_multiple_files(input_path)
        else:
            text = self.load_file(input_path)
        
        if not text:
            print("✗ Failed to load text")
            return None
        
        print(f"Initial size: {len(text):,} chars ({len(text)/(1024*1024):.2f} MB)")
        
        # Step 2: Remove metadata
        if remove_meta:
            print("\n[2/6] Removing metadata...")
            original_len = len(text)
            text = self.remove_metadata(text, aggressive=(clean_level == 'aggressive'))
            removed = original_len - len(text)
            print(f"Removed: {removed:,} chars ({removed/original_len*100:.1f}%)")
            print(f"After metadata removal: {len(text):,} chars")
        else:
            print("\n[2/6] Skipping metadata removal")
        
        # Step 3: Merge short lines into paragraphs
        if merge_lines:
            print("\n[3/6] Merging short lines into paragraphs...")
            original_lines = len(text.split('\n'))
            text = self.merge_short_lines(text)
            new_lines = len(text.split('\n'))
            print(f"Lines: {original_lines:,} → {new_lines:,} (merged {original_lines - new_lines:,})")
        else:
            print("\n[3/6] Skipping line merging")
        
        # Step 4: Clean
        print(f"\n[4/6] Cleaning text (level: {clean_level})...")
        original_len = len(text)
        text = self.clean_text(text, level=clean_level)
        removed = original_len - len(text)
        print(f"Removed: {removed:,} chars ({removed/original_len*100:.1f}%)")
        print(f"After cleaning: {len(text):,} chars")
        
        # Step 5: Filter charset
        if filter_charset:
            print("\n[5/6] Filtering character set...")
            original_len = len(text)
            text = self.filter_by_charset(
                text, 
                keep_cyrillic=True,
                keep_latin=keep_latin,
                keep_digits=True,
                keep_punctuation=True
            )
            removed = original_len - len(text)
            print(f"Removed: {removed:,} chars ({removed/original_len*100:.1f}%)")
            print(f"After filtering: {len(text):,} chars")
        else:
            print("\n[5/6] Skipping charset filtering")
        
        # Step 6: Validate and save
        print("\n[6/6] Validating and saving...")
        if not self.validate_dataset(text, min_size_mb=min_size_mb):
            print("✗ Dataset validation failed!")
            return None
        
        # Get statistics
        self.get_statistics(text, show_chars=True)
        
        # Save
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"✓ Dataset saved to: {output_path}")
            print(f"✓ Final size: {len(text):,} chars ({len(text)/(1024*1024):.2f} MB)")
            
        except Exception as e:
            print(f"✗ Failed to save dataset: {e}")
            return None
        
        self.text = text
        return text

if __name__ == "__main__":
    prep = DatasetPreparator()
    
    # Prepare from folder
    text = prep.prepare_dataset(
        input_path='..\\assets\\dataset\\numbers\\',
        output_path='..\\assets\\optimized\\dataset_numbers.txt',
        clean_level='medium',
        remove_meta=True,
        merge_lines=False,
        filter_charset=True,
        keep_latin=True,
        min_size_mb=0.01
    )
    
    if text:
        print("\n✓ Dataset preparation complete!")
    else:
        print("\n✗ Dataset preparation failed!")