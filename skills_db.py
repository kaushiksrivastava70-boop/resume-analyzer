"""Skills taxonomy and action verb dictionaries for resume evaluation."""

from typing import Dict, List, Set

# Comprehensive skills grouped by industry-standard technical categories
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "C", "Go",
        "Golang", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "R", "MATLAB",
        "Scala", "Dart", "Bash", "Shell", "PowerShell", "Perl", "Haskell", "Lua"
    ],
    "Web & Frameworks": [
        "React", "React.js", "Next.js", "Vue.js", "Vue", "Angular", "Node.js",
        "Express.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
        "Spring", "ASP.NET", ".NET", "Ruby on Rails", "Rails", "Laravel",
        "HTML5", "HTML", "CSS3", "CSS", "Sass", "SCSS", "Tailwind CSS",
        "Tailwind", "Bootstrap", "GraphQL", "RESTful API", "REST API", "REST",
        "WebSockets", "Redux", "Zustand", "Webpack", "Vite"
    ],
    "Data Science & AI/ML": [
        "Machine Learning", "Deep Learning", "Artificial Intelligence", "AI",
        "Data Science", "Data Analysis", "Natural Language Processing", "NLP",
        "Computer Vision", "Large Language Models", "LLMs", "Generative AI",
        "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Pandas", "NumPy",
        "SciPy", "Matplotlib", "Seaborn", "OpenCV", "Hugging Face", "LangChain",
        "LlamaIndex", "RAG", "Data Mining", "Feature Engineering",
        "Model Deployment", "MLOps", "Statistics", "Predictive Modeling",
        "Jupyter", "Tableau", "Power BI"
    ],
    "Cloud & DevOps": [
        "Amazon Web Services", "AWS", "Microsoft Azure", "Azure",
        "Google Cloud Platform", "GCP", "Google Cloud", "Docker", "Kubernetes",
        "K8s", "CI/CD", "Continuous Integration", "Continuous Deployment",
        "Jenkins", "GitHub Actions", "GitLab CI", "Terraform", "Ansible",
        "Linux", "Unix", "Nginx", "Apache", "Helm", "Prometheus", "Grafana",
        "Serverless", "AWS Lambda", "CloudFormation", "OpenTelemetry"
    ],
    "Databases & Tools": [
        "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra",
        "DynamoDB", "Elasticsearch", "Snowflake", "BigQuery", "Oracle",
        "Microsoft SQL Server", "SQL Server", "SQL", "NoSQL", "Supabase",
        "Firebase", "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
        "Kafka", "RabbitMQ", "Celery", "Postman", "Swagger"
    ],
    "Soft Skills & Leadership": [
        "Agile", "Scrum", "Kanban", "Cross-Functional Collaboration",
        "Leadership", "Team Leadership", "Mentorship", "Problem Solving",
        "Critical Thinking", "Communication", "Public Speaking",
        "Project Management", "Time Management", "Code Review", "System Design",
        "Strategic Planning", "Adaptability", "Collaboration", "Conflict Resolution",
        "Stakeholder Management"
    ]
}

# Strong impact action verbs frequently evaluated by ATS and technical recruiters
ACTION_VERBS: Set[str] = {
    # Achievement & Execution
    "achieved", "accomplished", "delivered", "executed", "completed", "produced",
    "exceeded", "surpassed", "attained", "awarded", "earned", "generated",

    # Leadership & Direction
    "led", "directed", "managed", "orchestrated", "spearheaded", "mentored",
    "guided", "trained", "supervised", "championed", "organized", "mobilized",
    "governed", "empowered",

    # Development & Architecture
    "built", "developed", "architected", "designed", "implemented", "engineered",
    "created", "constructed", "formulated", "programmed", "authored", "deployed",
    "integrated", "refactored", "migrated", "automated",

    # Optimization & Problem Solving
    "optimized", "enhanced", "streamlined", "accelerated", "reduced", "decreased",
    "eliminated", "resolved", "diagnosed", "troubleshot", "overhauled", "improved",
    "standardized", "boosted", "maximized", "transformed", "modernized",

    # Research & Strategy
    "analyzed", "evaluated", "researched", "investigated", "identified",
    "discovered", "benchmarked", "audited", "modeled", "forecasted", "formulated"
}

# Common resume section header keywords
SECTION_KEYWORDS: Dict[str, List[str]] = {
    "Contact Info": ["email", "phone", "linkedin", "github", "address", "portfolio"],
    "Summary / Objective": ["summary", "professional summary", "executive summary", "objective", "career objective", "about me", "profile"],
    "Work Experience": ["experience", "work experience", "employment", "professional experience", "work history", "career history"],
    "Education": ["education", "academic background", "academics", "qualifications", "degrees", "university", "college"],
    "Skills": ["skills", "technical skills", "core competencies", "technologies", "expertise", "proficiencies", "areas of expertise"],
    "Projects": ["projects", "personal projects", "academic projects", "key projects", "notable projects", "technical projects"],
    "Certifications": ["certifications", "licenses", "certificates", "credentials", "professional certifications", "awards", "honors"]
}
