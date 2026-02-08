# from Src.nlp.resume_parser import parse_resume

# result = parse_resume("sample_resume.pdf")
# print(result)


# import sys
# import os
# # from nlp.resume_parser import parse_resume

# result = parse_resume("Src/nlp/sample_resume.pdf")
# print(result)

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Src.nlp.resume_parser import parse_resume

# Path to sample resume
PDF_PATH = os.path.join(os.path.dirname(__file__), "sample_resume.pdf")

result = parse_resume(PDF_PATH)

print("\n=== Resume Analysis ===")
print("Skills found:", result["skills"])
print("Skill count:", result["skill_count"])
