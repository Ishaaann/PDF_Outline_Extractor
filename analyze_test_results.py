import json
import os

def analyze_test_results():
    """Analyze all test results and check for duplicates."""
    test_output_dir = "test_output"
    
    print("="*60)
    print("COMPREHENSIVE TEST RESULTS ANALYSIS")
    print("="*60)
    
    json_files = [f for f in os.listdir(test_output_dir) if f.endswith('.json')]
    
    for json_file in sorted(json_files):
        filepath = os.path.join(test_output_dir, json_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📄 File: {json_file}")
        print(f"   PDF: {json_file.replace('.json', '.pdf')}")
        print(f"   Title: '{data['title']}'")
        print(f"   Outline entries: {len(data['outline'])}")
        
        if data['outline']:
            print("   Outline structure:")
            for entry in data['outline']:
                print(f"     {entry['level']}: {entry['text']} (page {entry['page']})")
            
            # Check for duplicates
            texts = [item['text'] for item in data['outline']]
            unique_texts = set(texts)
            
            if len(texts) != len(unique_texts):
                duplicates = len(texts) - len(unique_texts)
                print(f"   ⚠️  {duplicates} duplicate(s) found!")
            else:
                print("   ✅ No duplicates found")
        else:
            print("   📝 No outline entries (simple document structure)")
        
        print("-" * 60)
    
    print(f"\n✅ Analysis complete for {len(json_files)} test file(s)")
    print("="*60)

if __name__ == "__main__":
    analyze_test_results()
