import json

# Load and display the test output
with open('test_output/2023248_IshaanRaj_CW_WeeklyLogs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*50)
print("TEST DOCUMENT ANALYSIS RESULTS")
print("="*50)
print(f"File: 2023248_IshaanRaj_CW_WeeklyLogs.pdf")
print(f"Title extracted: '{data['title']}'")
print(f"Outline entries found: {len(data['outline'])}")

if data['outline']:
    print("\nOutline structure:")
    for entry in data['outline']:
        print(f"  {entry['level']}: {entry['text']} (page {entry['page']})")
else:
    print("\nNo outline entries found.")
    print("This could indicate:")
    print("- The document doesn't have hierarchical headings")
    print("- The headings don't match the trained patterns")
    print("- Confidence thresholds weren't met")

print("="*50)
