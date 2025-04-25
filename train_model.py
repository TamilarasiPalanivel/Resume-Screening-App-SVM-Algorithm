import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import re

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

print("Loading expanded dataset...")
# Create a larger dataset with more examples for each category
data = pd.DataFrame({
    'resume_text': [
        # IT Examples (10)
        "Software Developer with 5 years experience in Python, JavaScript, and React. Developed web applications and APIs. BS in Computer Science.",
        "Network Engineer with CCNA certification. Experience configuring and maintaining Cisco networks, VPNs, and firewalls.",
        "Full Stack Developer proficient in MERN stack. Created responsive web applications and RESTful APIs. Strong knowledge of database design.",
        "Data Scientist with expertise in machine learning algorithms, Python, R, and SQL. PhD in Computer Science with focus on AI.",
        "DevOps Engineer experienced with AWS, Docker, Kubernetes, Jenkins, and CI/CD pipelines. Automated deployment processes.",
        "IT Support Specialist with 3 years experience troubleshooting hardware and software issues. CompTIA A+ certified.",
        "Front-end Developer skilled in HTML5, CSS3, JavaScript, and modern frameworks like React and Vue. Created responsive UIs.",
        "Database Administrator with expertise in SQL Server, Oracle, and PostgreSQL. Managed database security, performance, and backups.",
        "Systems Analyst with experience gathering requirements, designing solutions, and implementing IT systems for business needs.",
        "Cybersecurity Specialist with knowledge of network security, penetration testing, and vulnerability assessment. CISSP certified.",
        
        # Healthcare Examples (10)
        "Registered Nurse with 7 years experience in intensive care. BSN degree with certification in Advanced Cardiac Life Support.",
        "Physical Therapist helping patients recover from injuries. DPT degree with specialization in sports rehabilitation.",
        "Medical Laboratory Technician with experience in blood banking and clinical chemistry. Associate degree in Medical Laboratory Science.",
        "Pharmacist with PharmD degree. Experience in retail and hospital pharmacy settings. Knowledge of medication therapy management.",
        "Healthcare Administrator managing daily operations of medical facility. MBA in Healthcare Administration.",
        "Medical Assistant with duties including patient intake, vital signs, and assisting with examinations. Certified Medical Assistant.",
        "Radiologic Technologist performing X-rays and other diagnostic imaging procedures. Associate degree in Radiologic Technology.",
        "Dental Hygienist providing preventive dental care. Associate degree in Dental Hygiene with state licensure.",
        "Occupational Therapist helping patients develop skills for daily living. Master's degree in Occupational Therapy.",
        "Physician Assistant with experience in family medicine. Master's degree in Physician Assistant Studies with state licensure.",
        
        # Marketing Examples (10)
        "Marketing Manager with experience developing and implementing marketing strategies. MBA with focus on marketing.",
        "Social Media Specialist managing company presence across platforms. Created content that increased engagement by 40%.",
        "SEO Specialist optimizing websites for search engines. Experience with keyword research, on-page SEO, and link building.",
        "Content Writer creating blog posts, whitepapers, and marketing materials. Bachelor's degree in English.",
        "Brand Manager developing and maintaining brand identity. MBA with experience in consumer packaged goods.",
        "Digital Marketing Coordinator running PPC campaigns on Google and social media platforms. Google Ads certified.",
        "Public Relations Specialist with experience in media relations and crisis communication. Bachelor's degree in Public Relations.",
        "Market Research Analyst collecting and analyzing data on market conditions. Experience with survey design and statistical analysis.",
        "Email Marketing Specialist designing campaigns with 25% above-industry-average open rates. Experience with marketing automation.",
        "Product Marketing Manager with experience in product launches and positioning. MBA with 5 years in technology marketing."
    ],
    'category': ['IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',
                'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 
                'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing',
                'Marketing', 'Marketing', 'Marketing', 'Marketing', 'Marketing']
})

# Clean the resume text
print("Cleaning text...")
data['cleaned_resume'] = data['resume_text'].apply(lambda x: cleanResume(x))

# Prepare features and labels
print("Preparing features and labels...")
X = data['cleaned_resume']
y = data['category']

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

# Create TF-IDF features
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVC model
print("Training SVC model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import classification_report, accuracy_score
y_pred = clf.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model, vectorizer, and encoder
print("Saving models...")
pickle.dump(clf, open('clf.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(le, open('encoder.pkl', 'wb'))

print("Done! Models saved as clf.pkl, tfidf.pkl, and encoder.pkl")