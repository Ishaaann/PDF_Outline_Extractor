import json
import os

def check_duplicates():
    output_dir = "output"
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                texts = [item['text'] for item in data['outline']]
                unique_texts = set(texts)
                
                if len(texts) != len(unique_texts):
                    duplicates = len(texts) - len(unique_texts)
                    print(f"{filename}: {duplicates} duplicates found")
                    # Show which texts are duplicated
                    seen = set()
                    for text in texts:
                        if text in seen:
                            print(f"  Duplicate: {text}")
                        seen.add(text)
                else:
                    print(f"{filename}: No duplicates")

if __name__ == "__main__":
    check_duplicates()
