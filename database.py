import re
import sqlite3
from typing import List, Dict, Any, Generator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILENAME = "moocs.db"

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    return conn

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            provider TEXT,
            level TEXT,
            duration TEXT,
            tags TEXT,
            description TEXT,
            url TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS internships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT,
            location TEXT,
            stipend TEXT,
            tags TEXT,
            description TEXT,
            url TEXT
        )
    """)
    conn.commit()
    if conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 0:
        _seed_courses(conn)
    if conn.execute("SELECT COUNT(*) FROM internships").fetchone()[0] == 0:
        _seed_internships(conn)
    conn.commit()
    conn.close()
    print(f"[DB] '{DB_FILENAME}' ready.")


def _seed_courses(conn):
    courses = [
        # Python / Data Science
        ("Python for Everybody", "Coursera | University of Michigan", "Beginner", "8 Weeks",
         "Python programming basics loops functions beginner",
         "Learn Python from scratch with hands-on projects.",
         "https://www.coursera.org/specializations/python"),
        ("Applied Machine Learning in Python", "Coursera | University of Michigan", "Intermediate", "4 Weeks",
         "Python ML scikit-learn machine learning data",
         "Applied ML using Python and scikit-learn.",
         "https://www.coursera.org/learn/python-machine-learning"),
        ("Data Science Professional Certificate", "IBM | Coursera", "Beginner", "5 Months",
         "data science Python SQL statistics analysis",
         "End-to-end data science with SQL and Python.",
         "https://www.coursera.org/professional-certificates/ibm-data-science"),
        ("Data Analysis with Python", "freeCodeCamp", "Beginner", "3 Weeks",
         "Python data analysis pandas numpy matplotlib visualization",
         "Analyze datasets using pandas and numpy.",
         "https://www.freecodecamp.org/learn/data-analysis-with-python/"),
        ("Statistics for Data Science", "edX | MIT", "Intermediate", "6 Weeks",
         "statistics probability data science hypothesis testing",
         "Probability and statistics for data scientists.",
         "https://www.edx.org/course/statistics-and-data-science"),
        # ML / AI / Deep Learning
        ("Deep Learning Specialization", "DeepLearning.AI", "Advanced", "3 Months",
         "neural networks deep learning TensorFlow AI CNN RNN",
         "Master deep learning and neural networks.",
         "https://www.coursera.org/specializations/deep-learning"),
        ("Machine Learning Specialization", "Coursera | Stanford", "Intermediate", "3 Months",
         "machine learning regression classification supervised unsupervised",
         "Andrew Ng's complete ML course.",
         "https://www.coursera.org/specializations/machine-learning-introduction"),
        ("AI For Everyone", "DeepLearning.AI", "Beginner", "4 Weeks",
         "artificial intelligence AI business strategy non-technical",
         "Non-technical intro to AI and its business impact.",
         "https://www.coursera.org/learn/ai-for-everyone"),
        ("Computer Vision with TensorFlow", "Coursera", "Intermediate", "6 Weeks",
         "computer vision TensorFlow image recognition CNN object detection",
         "Build image classifiers with CNNs.",
         "https://www.coursera.org/learn/introduction-tensorflow"),
        ("NLP with Python", "fast.ai", "Advanced", "8 Weeks",
         "NLP natural language processing Python transformers BERT text",
         "NLP using modern transformer models.",
         "https://www.fast.ai/"),
        ("Generative AI Fundamentals", "Google Cloud | Coursera", "Intermediate", "2 Weeks",
         "generative AI LLM large language model prompt GPT ChatGPT",
         "Introduction to generative AI and LLMs.",
         "https://www.coursera.org/learn/introduction-to-generative-ai"),
        ("MLOps Specialization", "DeepLearning.AI", "Advanced", "4 Months",
         "MLOps machine learning production deployment pipeline monitoring",
         "Deploy and maintain ML systems in production.",
         "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops"),
        ("Reinforcement Learning Specialization", "Coursera | Alberta", "Advanced", "4 Months",
         "reinforcement learning RL reward policy agent Q-learning",
         "Learn RL from fundamentals to advanced.",
         "https://www.coursera.org/specializations/reinforcement-learning"),
        # Web Development
        ("Full Stack Web Development Bootcamp", "Udemy | Colt Steele", "Intermediate", "3 Months",
         "JavaScript React Node.js HTML CSS fullstack web development",
         "Build full stack web apps with React and Node.",
         "https://www.udemy.com/course/the-web-developer-bootcamp/"),
        ("React - The Complete Guide", "Udemy | Maximilian", "Intermediate", "6 Weeks",
         "React JavaScript frontend hooks components JSX Redux",
         "Master React with hooks and advanced patterns.",
         "https://www.udemy.com/course/react-the-complete-guide-incl-redux/"),
        ("Node.js Developer Course", "Udemy", "Intermediate", "5 Weeks",
         "Node.js backend JavaScript REST API Express MongoDB",
         "Build backends using Node and Express.",
         "https://www.udemy.com/course/the-complete-nodejs-developer-course-2/"),
        ("Next.js & React Complete Guide", "Udemy", "Intermediate", "5 Weeks",
         "Next.js React SSR fullstack JavaScript frontend",
         "Build production apps with Next.js.",
         "https://www.udemy.com/course/nextjs-react-the-complete-guide/"),
        ("HTML & CSS for Beginners", "freeCodeCamp", "Beginner", "2 Weeks",
         "HTML CSS web design frontend beginner layout",
         "Build your first webpage with HTML and CSS.",
         "https://www.freecodecamp.org/learn/responsive-web-design/"),
        ("Vue.js Complete Guide", "Udemy", "Intermediate", "4 Weeks",
         "Vue.js JavaScript frontend framework components reactive",
         "Build reactive UIs using Vue.js.",
         "https://www.udemy.com/course/vuejs-2-the-complete-guide/"),
        ("TypeScript Masterclass", "Udemy", "Intermediate", "3 Weeks",
         "TypeScript JavaScript types interfaces frontend static typing",
         "Learn TypeScript to write better JavaScript.",
         "https://www.udemy.com/course/understanding-typescript/"),
        # SQL / Databases
        ("SQL for Data Science", "Coursera | UC Davis", "Beginner", "4 Weeks",
         "SQL database queries data science analysis SELECT JOIN",
         "Learn SQL to query and analyze data.",
         "https://www.coursera.org/learn/sql-for-data-science"),
        ("The Complete SQL Bootcamp", "Udemy", "Beginner", "3 Weeks",
         "SQL PostgreSQL database queries joins aggregation",
         "Complete SQL from basics to advanced queries.",
         "https://www.udemy.com/course/the-complete-sql-bootcamp/"),
        ("MongoDB - The Complete Guide", "Udemy", "Intermediate", "5 Weeks",
         "MongoDB NoSQL database collections documents aggregation",
         "Master MongoDB for modern applications.",
         "https://www.udemy.com/course/mongodb-the-complete-developers-guide/"),
        ("Database Design and SQL", "Coursera", "Beginner", "4 Weeks",
         "database design SQL normalization schema relational",
         "Design relational databases and write SQL.",
         "https://www.coursera.org/learn/database-management"),
        # Cloud / DevOps
        ("Cloud Computing on AWS", "AWS Training", "Intermediate", "6 Weeks",
         "AWS cloud EC2 S3 Lambda DevOps infrastructure Amazon",
         "Deploy and manage apps on Amazon Web Services.",
         "https://www.coursera.org/learn/aws-cloud-technical-essentials"),
        ("Google Cloud Fundamentals", "Google Cloud | Coursera", "Beginner", "4 Weeks",
         "Google Cloud GCP cloud computing infrastructure BigQuery",
         "Get started with Google Cloud Platform.",
         "https://www.coursera.org/learn/gcp-fundamentals"),
        ("Docker and Kubernetes Bootcamp", "Udemy", "Intermediate", "6 Weeks",
         "Docker Kubernetes containers DevOps deployment orchestration",
         "Containerize and orchestrate applications.",
         "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"),
        ("DevOps Engineer Bootcamp", "LinkedIn Learning", "Intermediate", "8 Weeks",
         "DevOps CI/CD Jenkins GitHub pipelines automation infrastructure",
         "Build and automate DevOps pipelines.",
         "https://www.linkedin.com/learning/paths/become-a-devops-engineer"),
        ("Azure Fundamentals AZ-900", "Microsoft Learn", "Beginner", "4 Weeks",
         "Azure Microsoft cloud fundamentals services virtual machines",
         "Prepare for Azure AZ-900 certification.",
         "https://learn.microsoft.com/en-us/certifications/azure-fundamentals/"),
        ("Terraform for Beginners", "Udemy", "Intermediate", "3 Weeks",
         "Terraform infrastructure as code IaC cloud AWS provisioning",
         "Manage cloud infrastructure with Terraform.",
         "https://www.udemy.com/course/terraform-beginner-to-advanced/"),
        # Cybersecurity
        ("Cybersecurity Fundamentals", "Coursera | IBM", "Beginner", "2 Months",
         "cybersecurity security networking Linux firewall threats",
         "Introduction to cybersecurity concepts.",
         "https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst"),
        ("Ethical Hacking Bootcamp", "Udemy", "Intermediate", "8 Weeks",
         "ethical hacking penetration testing cybersecurity network vulnerability",
         "Learn ethical hacking and pen testing.",
         "https://www.udemy.com/course/learn-ethical-hacking-from-scratch/"),
        ("CompTIA Security+ Prep", "LinkedIn Learning", "Intermediate", "6 Weeks",
         "security CompTIA certification network threats cryptography",
         "Prepare for CompTIA Security+ exam.",
         "https://www.linkedin.com/learning/paths/prepare-for-the-comptia-security-plus-sy0-601-exam"),
        # Mobile
        ("iOS App Development with Swift", "Apple Developer | Udemy", "Intermediate", "8 Weeks",
         "iOS Swift mobile development Apple Xcode UIKit",
         "Build iOS apps using Swift and Xcode.",
         "https://www.udemy.com/course/ios-13-app-development-bootcamp/"),
        ("Android Development with Kotlin", "Google | Udacity", "Intermediate", "3 Months",
         "Android Kotlin mobile development app Java XML",
         "Build Android apps using Kotlin.",
         "https://www.udacity.com/course/developing-android-apps-with-kotlin--ud9012"),
        ("Flutter & Dart Complete Guide", "Udemy", "Intermediate", "6 Weeks",
         "Flutter Dart mobile iOS Android cross-platform widgets",
         "Build cross-platform apps with Flutter.",
         "https://www.udemy.com/course/learn-flutter-dart-to-build-ios-android-apps/"),
        ("React Native Mobile Apps", "Udemy", "Intermediate", "5 Weeks",
         "React Native JavaScript mobile iOS Android Expo",
         "Build mobile apps using React Native.",
         "https://www.udemy.com/course/the-complete-react-native-and-redux-course/"),
        # UI/UX
        ("Google UX Design Certificate", "Google | Coursera", "Beginner", "6 Months",
         "UI UX design Figma prototyping wireframe user experience research",
         "Google's professional UX design certificate.",
         "https://www.coursera.org/professional-certificates/google-ux-design"),
        ("Figma for UX Designers", "Udemy", "Beginner", "3 Weeks",
         "Figma UI design prototype wireframe layout components",
         "Design modern UIs with Figma.",
         "https://www.udemy.com/course/figma-ux-ui-design-user-experience-tutorial-course/"),
        # Data Engineering
        ("Data Engineering Professional Certificate", "IBM | Coursera", "Intermediate", "5 Months",
         "data engineering pipeline ETL Spark Hadoop SQL Kafka",
         "Build data pipelines and ETL workflows.",
         "https://www.coursera.org/professional-certificates/ibm-data-engineer"),
        ("Apache Spark with Python", "Udemy", "Advanced", "5 Weeks",
         "Apache Spark PySpark big data distributed data engineering Python",
         "Process big data with Spark and PySpark.",
         "https://www.udemy.com/course/apache-spark-with-python-big-data-with-pyspark-and-spark/"),
        ("Kafka for Beginners", "Udemy", "Intermediate", "3 Weeks",
         "Kafka streaming data pipeline messaging real-time events",
         "Learn event streaming with Apache Kafka.",
         "https://www.udemy.com/course/apache-kafka/"),
        # Algorithms / CS
        ("Data Structures and Algorithms", "edX | UC San Diego", "Beginner", "6 Weeks",
         "algorithms data structures arrays trees graphs sorting",
         "Core algorithms and data structures.",
         "https://www.edx.org/course/data-structures-fundamentals"),
        ("Algorithms Specialization", "Coursera | Stanford", "Advanced", "4 Months",
         "algorithms graph dynamic programming divide conquer NP",
         "Stanford's in-depth algorithms course.",
         "https://www.coursera.org/specializations/algorithms"),
        ("CS50: Intro to Computer Science", "edX | Harvard", "Beginner", "3 Months",
         "computer science programming C Python web fundamentals",
         "Harvard's famous intro to CS.",
         "https://www.edx.org/course/introduction-computer-science-harvardx-cs50x"),
        # Languages
        ("Java Programming Masterclass", "Udemy", "Beginner", "4 Months",
         "Java programming OOP object oriented classes Spring",
         "Complete Java from beginner to expert.",
         "https://www.udemy.com/course/java-the-complete-java-developer-course/"),
        ("C++ Programming Bootcamp", "Udemy", "Intermediate", "6 Weeks",
         "C++ programming pointers memory OOP systems performance",
         "Master C++ for systems programming.",
         "https://www.udemy.com/course/beginning-c-plus-plus-programming/"),
        ("Go Programming Language", "Udemy", "Intermediate", "4 Weeks",
         "Go Golang backend programming concurrency goroutines",
         "Learn Go for high-performance backends.",
         "https://www.udemy.com/course/go-the-complete-developers-guide/"),
        ("Rust Programming for Beginners", "Udemy", "Intermediate", "5 Weeks",
         "Rust programming systems memory safety ownership",
         "Build safe and fast programs with Rust.",
         "https://www.udemy.com/course/rust-lang/"),
        # Business / PM
        ("Project Management Professional PMP", "PMI | Coursera", "Intermediate", "3 Months",
         "project management PMP agile scrum leadership planning",
         "Prepare for PMP certification.",
         "https://www.coursera.org/professional-certificates/google-project-management"),
        ("Agile & Scrum Fundamentals", "LinkedIn Learning", "Beginner", "2 Weeks",
         "agile scrum sprint project management team ceremonies",
         "Learn agile methodology and Scrum.",
         "https://www.linkedin.com/learning/agile-foundations"),
        ("Digital Marketing Fundamentals", "Google Digital Garage", "Beginner", "4 Weeks",
         "digital marketing SEO social media email analytics campaigns",
         "Get started with digital marketing.",
         "https://learndigital.withgoogle.com/digitalgarage/course/digital-marketing"),
        ("Excel for Data Analysis", "Coursera | PwC", "Beginner", "3 Weeks",
         "Excel spreadsheet data analysis pivot charts formulas VLOOKUP",
         "Use Excel for business data analysis.",
         "https://www.coursera.org/learn/excel-data-analysis"),
        ("Blockchain Fundamentals", "edX | Berkeley", "Intermediate", "5 Weeks",
         "blockchain cryptocurrency Bitcoin Ethereum smart contracts Web3",
         "Understand blockchain technology.",
         "https://www.edx.org/course/blockchain-technology"),
        ("Prompt Engineering for LLMs", "DeepLearning.AI", "Beginner", "2 Weeks",
         "prompt engineering LLM ChatGPT GPT AI generative language",
         "Learn to write effective prompts for AI models.",
         "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/"),
        ("Embedded Systems Programming", "Coursera | EIT", "Advanced", "8 Weeks",
         "embedded systems C microcontroller hardware firmware Arduino",
         "Program embedded hardware with C.",
         "https://www.coursera.org/learn/introduction-embedded-systems"),
        ("Game Development with Unity", "Coursera | Michigan State", "Intermediate", "4 Months",
         "game development Unity C# 2D 3D physics gameplay",
         "Build 2D and 3D games using Unity.",
         "https://www.coursera.org/specializations/game-design-and-development"),
        ("Power BI for Beginners", "Udemy", "Beginner", "3 Weeks",
         "Power BI business intelligence data visualization dashboard reports",
         "Build dashboards with Power BI.",
         "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/"),
        ("Tableau Desktop Specialist", "Udemy", "Beginner", "4 Weeks",
         "Tableau data visualization dashboard analytics charts",
         "Create stunning data visualizations.",
         "https://www.udemy.com/course/tableau10/"),
        ("Cybersecurity Specialization", "Coursera | Maryland", "Advanced", "5 Months",
         "cybersecurity advanced cryptography network security software",
         "Advanced cybersecurity concepts.",
         "https://www.coursera.org/specializations/cyber-security"),
        ("Kotlin for Android Developers", "Udemy", "Intermediate", "5 Weeks",
         "Kotlin Android mobile development Jetpack Compose",
         "Build modern Android apps with Kotlin.",
         "https://www.udemy.com/course/kotlin-android-developer-masterclass/"),
    ]
    conn.executemany(
        "INSERT INTO courses (title,provider,level,duration,tags,description,url) VALUES (?,?,?,?,?,?,?)",
        courses
    )
    print(f"[DB] Seeded {len(courses)} courses.")


def _seed_internships(conn):
    internships = [
        # Data / ML / AI
        ("Data Science Intern", "Google", "Remote", "$5K–$7K/mo",
         "Python SQL machine learning data science pandas statistics",
         "Work on real-world data science and ML projects.",
         "https://www.linkedin.com/jobs/search/?keywords=Data+Science+Intern&f_E=1"),
        ("Machine Learning Intern", "Meta", "Menlo Park, CA", "Competitive",
         "PyTorch ML deep learning AI model training neural networks",
         "Research and deploy ML models at scale.",
         "https://www.linkedin.com/jobs/search/?keywords=Machine+Learning+Intern&f_E=1"),
        ("AI Research Intern", "OpenAI", "San Francisco, CA", "$6K/mo",
         "NLP transformers Python research BERT language model AI LLM",
         "Research large language model fine-tuning.",
         "https://www.linkedin.com/jobs/search/?keywords=AI+Research+Intern&f_E=1"),
        ("Data Analyst Intern", "Amazon", "Remote", "$4K/mo",
         "SQL data analysis Excel Python statistics reporting dashboard",
         "Analyze business data and build dashboards.",
         "https://www.linkedin.com/jobs/search/?keywords=Data+Analyst+Intern&f_E=1"),
        ("Business Intelligence Intern", "Microsoft", "Redmond, WA", "$4.5K/mo",
         "SQL Power BI data visualization business intelligence analytics",
         "Build BI dashboards and reports.",
         "https://www.linkedin.com/jobs/search/?keywords=Business+Intelligence+Intern&f_E=1"),
        ("MLOps Intern", "Nvidia", "Santa Clara, CA", "$5K/mo",
         "MLOps Python Docker Kubernetes machine learning deployment",
         "Automate ML model deployment pipelines.",
         "https://www.linkedin.com/jobs/search/?keywords=MLOps+Intern&f_E=1"),
        ("Data Engineering Intern", "Spotify", "Remote", "$4.5K/mo",
         "data engineering ETL Python Spark SQL pipeline Kafka",
         "Build and maintain data pipelines.",
         "https://www.linkedin.com/jobs/search/?keywords=Data+Engineering+Intern&f_E=1"),
        ("Computer Vision Intern", "Tesla", "Palo Alto, CA", "$6K/mo",
         "computer vision Python TensorFlow image recognition deep learning",
         "Work on autonomous driving vision systems.",
         "https://www.linkedin.com/jobs/search/?keywords=Computer+Vision+Intern&f_E=1"),
        # Software Engineering
        ("Software Engineering Intern", "Apple", "Cupertino, CA", "$6K/mo",
         "Python Java software engineering algorithms OOP backend systems",
         "Build and test software features.",
         "https://www.linkedin.com/jobs/search/?keywords=Software+Engineering+Intern&f_E=1"),
        ("Backend Developer Intern", "Stripe", "Remote", "$5K/mo",
         "backend Python Node.js REST API databases PostgreSQL microservices",
         "Build robust backend APIs.",
         "https://www.linkedin.com/jobs/search/?keywords=Backend+Developer+Intern&f_E=1"),
        ("Full Stack Intern", "StartUp AI", "Remote", "$4K/mo",
         "fullstack React Node.js JavaScript HTML CSS API frontend backend",
         "Build features across the full stack.",
         "https://www.linkedin.com/jobs/search/?keywords=Full+Stack+Intern&f_E=1"),
        ("Java Developer Intern", "Oracle", "Austin, TX", "$4.5K/mo",
         "Java Spring Boot backend REST API microservices enterprise",
         "Develop Java microservices for enterprise software.",
         "https://www.linkedin.com/jobs/search/?keywords=Java+Developer+Intern&f_E=1"),
        ("Go Developer Intern", "Uber", "Remote", "$5K/mo",
         "Go Golang backend microservices distributed systems concurrency",
         "Build high-performance backend services.",
         "https://www.linkedin.com/jobs/search/?keywords=Golang+Developer+Intern&f_E=1"),
        ("Python Developer Intern", "Dropbox", "Remote", "$4K/mo",
         "Python backend Flask Django REST API automation scripting",
         "Build backend services with Python.",
         "https://www.linkedin.com/jobs/search/?keywords=Python+Developer+Intern&f_E=1"),
        # Frontend / UI
        ("Frontend Developer Intern", "Shopify", "Remote", "$3.5K/mo",
         "React JavaScript CSS frontend UI UX HTML components responsive",
         "Build responsive UI components.",
         "https://www.linkedin.com/jobs/search/?keywords=Frontend+Developer+Intern&f_E=1"),
        ("UI/UX Design Intern", "Figma", "San Francisco, CA", "$4K/mo",
         "UI UX design Figma prototype wireframe user research usability",
         "Design and test user interfaces.",
         "https://www.linkedin.com/jobs/search/?keywords=UI+UX+Design+Intern&f_E=1"),
        ("React Developer Intern", "Airbnb", "Remote", "$4.5K/mo",
         "React JavaScript TypeScript frontend Next.js hooks Redux",
         "Build and improve React components.",
         "https://www.linkedin.com/jobs/search/?keywords=React+Developer+Intern&f_E=1"),
        # Mobile
        ("iOS Developer Intern", "Snap", "Los Angeles, CA", "$5K/mo",
         "iOS Swift Xcode mobile Apple development UIKit",
         "Develop features for the Snapchat iOS app.",
         "https://www.linkedin.com/jobs/search/?keywords=iOS+Developer+Intern&f_E=1"),
        ("Android Developer Intern", "Google", "Remote", "$5K/mo",
         "Android Kotlin Java mobile development app XML Jetpack",
         "Build Android features for Google apps.",
         "https://www.linkedin.com/jobs/search/?keywords=Android+Developer+Intern&f_E=1"),
        ("Flutter Developer Intern", "Alibaba", "Remote", "$3.5K/mo",
         "Flutter Dart mobile cross-platform iOS Android widgets",
         "Build cross-platform mobile apps with Flutter.",
         "https://www.linkedin.com/jobs/search/?keywords=Flutter+Developer+Intern&f_E=1"),
        # Cloud / DevOps
        ("Cloud Engineer Intern", "Microsoft Azure", "Remote", "$4.5K/mo",
         "Azure cloud DevOps infrastructure Kubernetes Terraform",
         "Deploy cloud-native applications.",
         "https://www.linkedin.com/jobs/search/?keywords=Cloud+Engineer+Intern&f_E=1"),
        ("DevOps Intern", "IBM", "Remote", "$4K/mo",
         "DevOps CI/CD Docker Jenkins GitHub pipelines automation Linux",
         "Automate software delivery pipelines.",
         "https://www.linkedin.com/jobs/search/?keywords=DevOps+Intern&f_E=1"),
        ("AWS Cloud Intern", "Amazon Web Services", "Seattle, WA", "$5K/mo",
         "AWS cloud Lambda EC2 S3 infrastructure DevOps CDK",
         "Work on cloud infrastructure at scale.",
         "https://www.linkedin.com/jobs/search/?keywords=AWS+Cloud+Intern&f_E=1"),
        ("Site Reliability Intern", "LinkedIn", "Remote", "$5K/mo",
         "SRE reliability DevOps Python Linux monitoring Kubernetes",
         "Improve reliability of production systems.",
         "https://www.linkedin.com/jobs/search/?keywords=Site+Reliability+Intern&f_E=1"),
        # Cybersecurity
        ("Cybersecurity Intern", "Palo Alto Networks", "Remote", "$4.5K/mo",
         "cybersecurity security network firewall threat analysis SIEM",
         "Analyze and respond to security threats.",
         "https://www.linkedin.com/jobs/search/?keywords=Cybersecurity+Intern&f_E=1"),
        ("Penetration Testing Intern", "CrowdStrike", "Remote", "$4K/mo",
         "penetration testing ethical hacking cybersecurity vulnerability",
         "Find vulnerabilities in client systems.",
         "https://www.linkedin.com/jobs/search/?keywords=Penetration+Testing+Intern&f_E=1"),
        # Database
        ("Database Engineer Intern", "MongoDB Inc.", "Remote", "$4K/mo",
         "MongoDB NoSQL database queries aggregation Atlas cloud",
         "Work on MongoDB core database features.",
         "https://www.linkedin.com/jobs/search/?keywords=Database+Engineer+Intern&f_E=1"),
        ("SQL Developer Intern", "Salesforce", "Remote", "$4K/mo",
         "SQL database PostgreSQL data modeling queries joins optimization",
         "Build and optimize SQL queries.",
         "https://www.linkedin.com/jobs/search/?keywords=SQL+Developer+Intern&f_E=1"),
        # Blockchain
        ("Blockchain Developer Intern", "Coinbase", "San Francisco, CA", "$5K/mo",
         "blockchain Ethereum Solidity smart contracts Web3 crypto DeFi",
         "Develop decentralized applications.",
         "https://www.linkedin.com/jobs/search/?keywords=Blockchain+Developer+Intern&f_E=1"),
        # PM / Business
        ("Product Manager Intern", "Google", "Mountain View, CA", "$6K/mo",
         "product management agile scrum roadmap analytics leadership strategy",
         "Drive product strategy and execution.",
         "https://www.linkedin.com/jobs/search/?keywords=Product+Manager+Intern&f_E=1"),
        ("Scrum Master Intern", "Accenture", "Remote", "$3.5K/mo",
         "agile scrum project management sprint planning team ceremonies",
         "Facilitate agile ceremonies and sprints.",
         "https://www.linkedin.com/jobs/search/?keywords=Scrum+Master+Intern&f_E=1"),
        ("Digital Marketing Intern", "HubSpot", "Remote", "$3K/mo",
         "digital marketing SEO content analytics social media email",
         "Run digital marketing campaigns.",
         "https://www.linkedin.com/jobs/search/?keywords=Digital+Marketing+Intern&f_E=1"),
        ("Business Analyst Intern", "Deloitte", "Remote", "$4K/mo",
         "business analysis SQL Excel data reporting stakeholder requirements",
         "Analyze business processes and data.",
         "https://www.linkedin.com/jobs/search/?keywords=Business+Analyst+Intern&f_E=1"),
        # Other
        ("Prompt Engineer Intern", "Anthropic", "San Francisco, CA", "$5K/mo",
         "prompt engineering LLM GPT AI generative language model NLP",
         "Design and test prompts for LLMs.",
         "https://www.linkedin.com/jobs/search/?keywords=Prompt+Engineer+Intern&f_E=1"),
        ("Game Developer Intern", "EA Sports", "Remote", "$4K/mo",
         "game development Unity C# Python graphics gameplay 3D",
         "Build and test gameplay features.",
         "https://www.linkedin.com/jobs/search/?keywords=Game+Developer+Intern&f_E=1"),
        ("Embedded Systems Intern", "Qualcomm", "San Diego, CA", "$5K/mo",
         "embedded systems C C++ hardware firmware microcontroller IoT",
         "Develop firmware for embedded hardware.",
         "https://www.linkedin.com/jobs/search/?keywords=Embedded+Systems+Intern&f_E=1"),
    ]
    conn.executemany(
        "INSERT INTO internships (title,company,location,stipend,tags,description,url) VALUES (?,?,?,?,?,?,?)",
        internships
    )
    print(f"[DB] Seeded {len(internships)} internships.")


# ─────────────────────────────────────────────
# USER AUTH
# ─────────────────────────────────────────────
def get_user_by_email(email: str):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(name: str, email: str, password_hash: str):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, password_hash))
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        return dict(user)
    except sqlite3.IntegrityError:
        conn.close()
        return None


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
STOP_WORDS = {"and", "or", "the", "a", "an", "is", "in", "to", "for", "with", "on", "of"}

def stream_tokenize(text: str) -> Generator[str, None, None]:
    for match in re.finditer(r'\b[a-zA-Z]{2,}\b', text):
        token = match.group(0).lower()
        if token not in STOP_WORDS:
            yield token

def map_seniority(role: str) -> str:
    role_lower = role.lower()
    if any(k in role_lower for k in ["senior", "lead", "advanced", "architect", "manager", "principal"]):
        return "Advanced"
    elif any(k in role_lower for k in ["junior", "student", "intern", "beginner", "entry", "fresher"]):
        return "Beginner"
    return "Intermediate"

def get_level_score(item_level: str, mapped_level: str) -> float:
    levels = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    if item_level not in levels or mapped_level not in levels:
        return 0.5
    diff = abs(levels[item_level] - levels[mapped_level])
    return 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)

def heuristic_estimation(user_tokens: set, item_text: str) -> float:
    item_tokens = set(stream_tokenize(item_text))
    if not item_tokens:
        return 0.0
    return len(user_tokens.intersection(item_tokens)) / max(len(user_tokens), 1)

def content_based_similarity(user_text: str, item_texts: List[str]) -> List[float]:
    if not item_texts:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([user_text] + item_texts)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten().tolist()
    except ValueError:
        return [0.0] * len(item_texts)

def rank_items(user_skills: str, user_role: str, items: List[Dict[str, Any]], item_type: str = "course") -> List[Dict[str, Any]]:
    user_text = f"{user_skills} {user_role}"
    user_tokens = set(stream_tokenize(user_text))
    mapped_level = map_seniority(user_role)
    item_texts = [f"{i['title']} {i['tags']} {i.get('description', '')}" for i in items]
    content_scores = content_based_similarity(user_text, item_texts)
    results = []
    for idx, item in enumerate(items):
        h = heuristic_estimation(user_tokens, item_texts[idx])
        c = content_scores[idx]
        l = get_level_score(item.get('level', ''), mapped_level) if item_type == "course" else 0.8
        score = (c * 0.5) + (h * 0.3) + (l * 0.2)
        copy = item.copy()
        copy['match'] = min(100, max(0, int(score * 100)))
        results.append(copy)
    return sorted(results, key=lambda x: x['match'], reverse=True)
