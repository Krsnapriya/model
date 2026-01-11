import google.generativeai as genai

genai.configure(api_key="AIzaSyA2JAf-osZjk0KI5bLtMPFtC7AjEv9FP04")

print("Listing models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
