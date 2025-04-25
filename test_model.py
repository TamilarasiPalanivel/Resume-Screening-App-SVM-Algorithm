import pickle
import re

# Load the saved models
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Test examples
test_resumes = [
    "Python developer with 3 years experience building web applications using Django and Flask",
    "Registered nurse with experience in emergency room and patient care",
    "Marketing specialist with experience in social media campaigns and content creation"
]

# Predict category for each test resume
print("Testing model predictions:")
print("--------------------------")
for resume in test_resumes:
    cleaned = cleanResume(resume)
    vectorized = tfidf.transform([cleaned])
    prediction = svc_model.predict(vectorized)
    category = le.inverse_transform(prediction)[0]
    print(f"Resume: {resume[:50]}...")
    print(f"Prediction: {category}\n")