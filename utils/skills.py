"""Curated, dependency-free skill taxonomy and matcher.

Resume NER models are trained on news text and miss most technology names, so
skill detection is done with a hand-maintained alias table (≈200 skills) plus
word-boundary regexes. Ambiguous one-word skills (Go, Swift, Excel, Spark …)
are matched case-sensitively to avoid false positives such as "go to market".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Taxonomy:  canonical -> (category, aliases)  -- case-insensitive by default
# ---------------------------------------------------------------------------

_RAW: dict[str, tuple[str, tuple[str, ...]]] = {
    # ---------------- Languages ----------------
    "Python": ("Languages", ("python", "python3")),
    "JavaScript": ("Languages", ("javascript", "js", "ecmascript")),
    "TypeScript": ("Languages", ("typescript",)),
    "Java": ("Languages", ("java",)),
    "C++": ("Languages", ("c++", "cpp")),
    "C#": ("Languages", ("c#", "csharp", "c sharp")),
    "Go": ("Languages", ("go", "golang")),
    "Rust": ("Languages", ("rust",)),
    "Ruby": ("Languages", ("ruby",)),
    "PHP": ("Languages", ("php",)),
    "Swift": ("Languages", ("swift",)),
    "Kotlin": ("Languages", ("kotlin",)),
    "Scala": ("Languages", ("scala",)),
    "R": ("Languages", ("r programming", "r language", "rstudio")),
    "MATLAB": ("Languages", ("matlab",)),
    "Perl": ("Languages", ("perl",)),
    "Bash": ("Languages", ("bash", "shell scripting", "shell script", "zsh")),
    "PowerShell": ("Languages", ("powershell", "power shell")),
    "SQL": ("Languages", ("sql",)),
    "HTML": ("Languages", ("html", "html5")),
    "CSS": ("Languages", ("css", "css3")),
    "Sass": ("Languages", ("sass", "scss")),
    "Solidity": ("Languages", ("solidity",)),
    "Dart": ("Languages", ("dart",)),
    "Elixir": ("Languages", ("elixir",)),
    "Haskell": ("Languages", ("haskell",)),
    "Lua": ("Languages", ("lua",)),
    "Assembly": ("Languages", ("assembly", "x86 assembly", "arm assembly")),
    "VBA": ("Languages", ("vba", "visual basic")),
    "Groovy": ("Languages", ("groovy",)),
    "Julia": ("Languages", ("julia",)),
    "Clojure": ("Languages", ("clojure",)),
    "Fortran": ("Languages", ("fortran",)),
    "COBOL": ("Languages", ("cobol",)),
    # ---------------- Frontend ----------------
    "React": ("Frontend", ("react", "react.js", "reactjs")),
    "Next.js": ("Frontend", ("next.js", "nextjs", "next js")),
    "Vue.js": ("Frontend", ("vue.js", "vuejs", "vue")),
    "Angular": ("Frontend", ("angular", "angularjs")),
    "Svelte": ("Frontend", ("svelte", "sveltekit")),
    "Redux": ("Frontend", ("redux", "redux toolkit")),
    "Tailwind CSS": ("Frontend", ("tailwind css", "tailwindcss", "tailwind")),
    "Bootstrap": ("Frontend", ("bootstrap",)),
    "Material UI": ("Frontend", ("material ui", "mui", "material design")),
    "jQuery": ("Frontend", ("jquery",)),
    "Webpack": ("Frontend", ("webpack",)),
    "Vite": ("Frontend", ("vite", "vitejs")),
    "Storybook": ("Frontend", ("storybook",)),
    "Three.js": ("Frontend", ("three.js", "threejs")),
    "WebGL": ("Frontend", ("webgl",)),
    "Accessibility": ("Frontend", ("accessibility", "a11y", "wcag", "aria")),
    "Responsive Design": ("Frontend", ("responsive design", "responsive web design")),
    # ---------------- Backend ----------------
    "Node.js": ("Backend", ("node.js", "nodejs", "node js")),
    "Express.js": ("Backend", ("express.js", "expressjs", "express js")),
    "NestJS": ("Backend", ("nestjs", "nest.js")),
    "Django": ("Backend", ("django", "django rest framework", "drf")),
    "Flask": ("Backend", ("flask",)),
    "FastAPI": ("Backend", ("fastapi", "fast api")),
    "Spring Boot": ("Backend", ("spring boot", "springboot", "spring framework", "spring mvc")),
    "ASP.NET": ("Backend", ("asp.net", "aspnet", ".net core", "dotnet")),
    "Laravel": ("Backend", ("laravel",)),
    "Ruby on Rails": ("Backend", ("ruby on rails", "rails")),
    "GraphQL": ("Backend", ("graphql", "apollo")),
    "REST API": ("Backend", ("rest api", "restful api", "rest apis", "restful services")),
    "gRPC": ("Backend", ("grpc", "protocol buffers", "protobuf")),
    "WebSocket": ("Backend", ("websocket", "websockets", "socket.io")),
    "Microservices": ("Backend", ("microservices", "microservice architecture")),
    "Celery": ("Backend", ("celery",)),
    "RabbitMQ": ("Backend", ("rabbitmq",)),
    "Apache Kafka": ("Backend", ("apache kafka", "kafka")),
    "Nginx": ("Backend", ("nginx",)),
    # ---------------- Mobile ----------------
    "React Native": ("Mobile", ("react native",)),
    "Flutter": ("Mobile", ("flutter",)),
    "SwiftUI": ("Mobile", ("swiftui",)),
    "Jetpack Compose": ("Mobile", ("jetpack compose", "compose ui")),
    "Android": ("Mobile", ("android", "android sdk")),
    "iOS": ("Mobile", ("ios", "ios sdk")),
    "Xamarin": ("Mobile", ("xamarin", "maui")),
    "Ionic": ("Mobile", ("ionic",)),
    "Expo": ("Mobile", ("expo",)),
    # ---------------- Data & ML ----------------
    "Machine Learning": ("Data & ML", ("machine learning", "ml models", "supervised learning")),
    "Deep Learning": ("Data & ML", ("deep learning", "neural networks")),
    "NLP": ("Data & ML", ("nlp", "natural language processing")),
    "Computer Vision": ("Data & ML", ("computer vision", "image classification")),
    "LLM": ("Data & ML", ("llm", "llms", "large language model", "large language models")),
    "RAG": ("Data & ML", ("rag", "retrieval augmented generation", "retrieval-augmented generation")),
    "PyTorch": ("Data & ML", ("pytorch", "torch")),
    "TensorFlow": ("Data & ML", ("tensorflow",)),
    "Keras": ("Data & ML", ("keras",)),
    "scikit-learn": ("Data & ML", ("scikit-learn", "sklearn", "scikit learn")),
    "XGBoost": ("Data & ML", ("xgboost", "xg boost")),
    "LightGBM": ("Data & ML", ("lightgbm",)),
    "CatBoost": ("Data & ML", ("catboost",)),
    "Hugging Face Transformers": ("Data & ML", ("hugging face", "huggingface", "transformers")),
    "LangChain": ("Data & ML", ("langchain", "lang chain")),
    "LlamaIndex": ("Data & ML", ("llamaindex", "llama index")),
    "Pandas": ("Data & ML", ("pandas",)),
    "NumPy": ("Data & ML", ("numpy",)),
    "SciPy": ("Data & ML", ("scipy",)),
    "Matplotlib": ("Data & ML", ("matplotlib",)),
    "Seaborn": ("Data & ML", ("seaborn",)),
    "Plotly": ("Data & ML", ("plotly",)),
    "OpenCV": ("Data & ML", ("opencv", "open cv")),
    "Apache Spark": ("Data & ML", ("apache spark", "pyspark", "spark")),
    "Hadoop": ("Data & ML", ("hadoop", "hdfs")),
    "Airflow": ("Data & ML", ("airflow", "apache airflow")),
    "dbt": ("Data & ML", ("dbt", "data build tool")),
    "ETL": ("Data & ML", ("etl", "elt", "data pipelines", "data pipeline")),
    "Data Warehousing": ("Data & ML", ("data warehousing", "data warehouse")),
    "Data Modeling": ("Data & ML", ("data modeling", "data modelling")),
    "Feature Engineering": ("Data & ML", ("feature engineering",)),
    "Statistics": ("Data & ML", ("statistics", "statistical analysis", "statistical modeling")),
    "A/B Testing": ("Data & ML", ("a/b testing", "ab testing", "split testing")),
    "Time Series Analysis": ("Data & ML", ("time series", "time-series analysis")),
    "Reinforcement Learning": ("Data & ML", ("reinforcement learning",)),
    "MLOps": ("Data & ML", ("mlops", "ml ops", "model monitoring")),
    "Model Deployment": ("Data & ML", ("model deployment", "model serving", "inference optimization")),
    "Prompt Engineering": ("Data & ML", ("prompt engineering", "prompt design")),
    "Vector Databases": ("Data & ML", ("vector database", "vector databases", "vector db", "vector store")),
    "Data Visualization": ("Data & ML", ("data visualization", "data visualisation", "dashboards")),
    "Data Analysis": ("Data & ML", ("data analysis", "data analytics", "exploratory data analysis")),
    "Data Engineering": ("Data & ML", ("data engineering", "data engineer")),
    "Big Data": ("Data & ML", ("big data",)),
    # ---------------- Databases ----------------
    "PostgreSQL": ("Databases", ("postgresql", "postgres")),
    "MySQL": ("Databases", ("mysql",)),
    "SQLite": ("Databases", ("sqlite",)),
    "MongoDB": ("Databases", ("mongodb", "mongo")),
    "DynamoDB": ("Databases", ("dynamodb", "dynamo db")),
    "Cassandra": ("Databases", ("cassandra",)),
    "Oracle Database": ("Databases", ("oracle database", "oracle db", "oracle")),
    "Microsoft SQL Server": ("Databases", ("sql server", "mssql", "microsoft sql server")),
    "MariaDB": ("Databases", ("mariadb",)),
    "Neo4j": ("Databases", ("neo4j",)),
    "Redis": ("Databases", ("redis",)),
    "Elasticsearch": ("Databases", ("elasticsearch", "elastic search")),
    "OpenSearch": ("Databases", ("opensearch",)),
    "Snowflake": ("Databases", ("snowflake",)),
    "BigQuery": ("Databases", ("bigquery", "big query")),
    "Redshift": ("Databases", ("redshift", "amazon redshift")),
    "Databricks": ("Databases", ("databricks",)),
    "Pinecone": ("Databases", ("pinecone",)),
    "pgvector": ("Databases", ("pgvector",)),
    "ChromaDB": ("Databases", ("chromadb", "chroma db")),
    "Firebase": ("Databases", ("firebase", "firestore")),
    # ---------------- Cloud & DevOps ----------------
    "AWS": ("Cloud & DevOps", ("aws", "amazon web services")),
    "Azure": ("Cloud & DevOps", ("azure", "microsoft azure")),
    "GCP": ("Cloud & DevOps", ("gcp", "google cloud platform", "google cloud")),
    "Docker": ("Cloud & DevOps", ("docker", "dockerfile", "docker compose")),
    "Kubernetes": ("Cloud & DevOps", ("kubernetes", "k8s")),
    "Terraform": ("Cloud & DevOps", ("terraform", "hashicorp terraform")),
    "Infrastructure as Code": ("Cloud & DevOps", ("infrastructure as code", "iac")),
    "Ansible": ("Cloud & DevOps", ("ansible",)),
    "Jenkins": ("Cloud & DevOps", ("jenkins",)),
    "GitHub Actions": ("Cloud & DevOps", ("github actions", "gh actions")),
    "GitLab CI": ("Cloud & DevOps", ("gitlab ci", "gitlab ci/cd")),
    "CircleCI": ("Cloud & DevOps", ("circleci", "circle ci")),
    "Helm": ("Cloud & DevOps", ("helm",)),
    "Prometheus": ("Cloud & DevOps", ("prometheus",)),
    "Grafana": ("Cloud & DevOps", ("grafana",)),
    "Datadog": ("Cloud & DevOps", ("datadog",)),
    "CloudFormation": ("Cloud & DevOps", ("cloudformation", "aws cloudformation")),
    "Pulumi": ("Cloud & DevOps", ("pulumi",)),
    "Serverless": ("Cloud & DevOps", ("serverless", "serverless architecture")),
    "AWS Lambda": ("Cloud & DevOps", ("aws lambda",)),
    "Amazon S3": ("Cloud & DevOps", ("amazon s3", "aws s3", "s3")),
    "EC2": ("Cloud & DevOps", ("ec2", "aws ec2")),
    "EKS": ("Cloud & DevOps", ("eks", "aws eks")),
    "Cloud Run": ("Cloud & DevOps", ("cloud run", "gcp cloud run")),
    "CI/CD": ("Cloud & DevOps", ("ci/cd", "cicd", "continuous integration", "continuous delivery", "continuous deployment")),
    "Linux": ("Cloud & DevOps", ("linux", "ubuntu", "unix")),
    "Git": ("Cloud & DevOps", ("git", "version control")),
    "Observability": ("Cloud & DevOps", ("observability", "monitoring", "distributed tracing")),
    "Site Reliability Engineering": ("Cloud & DevOps", ("site reliability engineering", "sre")),
    # ---------------- Security ----------------
    "Penetration Testing": ("Security", ("penetration testing", "pentesting", "pen testing", "ethical hacking")),
    "SIEM": ("Security", ("siem", "security information and event management")),
    "SOC": ("Security", ("soc", "security operations center")),
    "Splunk": ("Security", ("splunk",)),
    "OWASP": ("Security", ("owasp", "owasp top 10")),
    "NIST": ("Security", ("nist", "nist 800-53", "nist csf")),
    "ISO 27001": ("Security", ("iso 27001", "iso27001")),
    "Cryptography": ("Security", ("cryptography", "encryption")),
    "Incident Response": ("Security", ("incident response", "dfir", "digital forensics")),
    "Threat Modeling": ("Security", ("threat modeling", "threat modelling")),
    "Burp Suite": ("Security", ("burp suite", "burp")),
    "Wireshark": ("Security", ("wireshark",)),
    "Metasploit": ("Security", ("metasploit",)),
    "Nmap": ("Security", ("nmap",)),
    "Vulnerability Assessment": ("Security", ("vulnerability assessment", "vulnerability management", "vapt")),
    "Zero Trust": ("Security", ("zero trust", "ztna")),
    "IAM": ("Security", ("iam", "identity and access management")),
    "OAuth": ("Security", ("oauth", "oauth2", "openid connect", "oidc")),
    "JWT": ("Security", ("jwt", "json web token")),
    "TLS": ("Security", ("tls", "ssl")),
    # ---------------- Testing ----------------
    "Unit Testing": ("Testing", ("unit testing", "unit tests")),
    "Integration Testing": ("Testing", ("integration testing", "integration tests")),
    "PyTest": ("Testing", ("pytest", "py.test")),
    "JUnit": ("Testing", ("junit",)),
    "Selenium": ("Testing", ("selenium", "webdriver")),
    "Cypress": ("Testing", ("cypress",)),
    "Playwright": ("Testing", ("playwright",)),
    "Jest": ("Testing", ("jest",)),
    "Postman": ("Testing", ("postman",)),
    "Load Testing": ("Testing", ("load testing", "performance testing", "jmeter", "locust")),
    "Test Automation": ("Testing", ("test automation", "automated testing", "automation testing")),
    "TDD": ("Testing", ("tdd", "test-driven development")),
    # ---------------- Design & Product ----------------
    "Figma": ("Design & Product", ("figma",)),
    "UI/UX Design": ("Design & Product", ("ui/ux", "ui/ux design", "ux design", "ui design", "user experience", "user interface design")),
    "Wireframing": ("Design & Product", ("wireframing", "wireframes")),
    "Prototyping": ("Design & Product", ("prototyping", "prototypes")),
    "Adobe XD": ("Design & Product", ("adobe xd",)),
    "Photoshop": ("Design & Product", ("photoshop", "adobe photoshop")),
    "Illustrator": ("Design & Product", ("illustrator", "adobe illustrator")),
    "Design Systems": ("Design & Product", ("design systems", "design system")),
    "User Research": ("Design & Product", ("user research", "usability testing")),
    "Product Management": ("Design & Product", ("product management", "product owner")),
    "Roadmapping": ("Design & Product", ("roadmapping", "product roadmap")),
    # ---------------- Business & Analytics ----------------
    "Tableau": ("Business & Analytics", ("tableau",)),
    "Power BI": ("Business & Analytics", ("power bi", "powerbi")),
    "Looker": ("Business & Analytics", ("looker", "looker studio")),
    "Google Analytics": ("Business & Analytics", ("google analytics", "ga4")),
    "Excel": ("Business & Analytics", ("excel", "ms excel", "microsoft excel")),
    "Salesforce": ("Business & Analytics", ("salesforce", "sfdc")),
    "SAP": ("Business & Analytics", ("sap",)),
    "HubSpot": ("Business & Analytics", ("hubspot",)),
    "SEO": ("Business & Analytics", ("seo", "search engine optimization")),
    "Financial Modeling": ("Business & Analytics", ("financial modeling", "financial modelling")),
    "Market Research": ("Business & Analytics", ("market research",)),
    "CRM": ("Business & Analytics", ("crm", "customer relationship management")),
    "Supply Chain": ("Business & Analytics", ("supply chain", "logistics")),
    # ---------------- Methodologies ----------------
    "Agile": ("Methodologies", ("agile", "agile methodologies", "agile development")),
    "Scrum": ("Methodologies", ("scrum",)),
    "Kanban": ("Methodologies", ("kanban",)),
    "Jira": ("Methodologies", ("jira",)),
    "Confluence": ("Methodologies", ("confluence",)),
    "DevOps": ("Methodologies", ("devops",)),
    "Code Review": ("Methodologies", ("code review", "code reviews", "peer review")),
    "Technical Documentation": ("Methodologies", ("technical documentation", "documentation")),
    # ---------------- Soft Skills ----------------
    "Leadership": ("Soft Skills", ("leadership", "team leadership")),
    "Communication": ("Soft Skills", ("communication", "communication skills")),
    "Teamwork": ("Soft Skills", ("teamwork", "team player")),
    "Problem Solving": ("Soft Skills", ("problem solving", "problem-solving")),
    "Stakeholder Management": ("Soft Skills", ("stakeholder management", "stakeholders")),
    "Project Management": ("Soft Skills", ("project management", "program management")),
    "Mentoring": ("Soft Skills", ("mentoring", "mentorship")),
    "Collaboration": ("Soft Skills", ("collaboration", "cross-functional")),
    "Time Management": ("Soft Skills", ("time management",)),
    "Critical Thinking": ("Soft Skills", ("critical thinking",)),
    "Presentation": ("Soft Skills", ("presentation", "presentations", "public speaking")),
    "Negotiation": ("Soft Skills", ("negotiation",)),
    "Adaptability": ("Soft Skills", ("adaptability", "adaptable")),
    "Ownership": ("Soft Skills", ("ownership", "self-starter")),
}

# Skills whose bare token is a common English word; match case-sensitively.
_CASE_SENSITIVE = {"Go", "Swift", "Dart", "Excel", "RAG"}
# These need a disambiguating multi-word alias rather than the bare token.
_CASE_SENSITIVE_ALIASES = {
    "Go": ("Go", "Golang", "GOLANG"),
    "Swift": ("Swift",),
    "Dart": ("Dart",),
    "Apache Spark": ("Spark", "Apache Spark", "PySpark"),
    "Excel": ("Excel", "MS Excel", "Microsoft Excel"),
    "RAG": ("RAG",),
}

CATEGORIES = tuple(dict.fromkeys(category for category, _ in _RAW.values()))

_SKILL_ORDER = {name: index for index, name in enumerate(_RAW)}


@dataclass(frozen=True)
class SkillHit:
    name: str
    category: str
    occurrences: int
    in_skills_section: bool
    confidence: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "occurrences": self.occurrences,
            "in_skills_section": self.in_skills_section,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------

def _alias_pattern(alias: str, *, case_sensitive: bool) -> re.Pattern[str]:
    escaped = re.escape(alias)
    escaped = escaped.replace(r"\ ", r"[\s\-_]+")
    prefix = r"(?<![A-Za-z0-9])" if alias[0].isalnum() else r"(?<!\w)"
    suffix = r"(?![A-Za-z])" if alias[-1].isalnum() else r"(?![A-Za-z])"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(prefix + escaped + suffix, flags)


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...]:
    entries = []
    for name, (category, aliases) in _RAW.items():
        case_sensitive = name in _CASE_SENSITIVE
        effective = _CASE_SENSITIVE_ALIASES.get(name, aliases)
        patterns = tuple(_alias_pattern(alias, case_sensitive=case_sensitive) for alias in effective)
        entries.append((name, category, patterns))
    return tuple(entries)


@lru_cache(maxsize=1)
def alias_index() -> dict[str, str]:
    """Lower-cased alias -> canonical name, for fuzzy/known checks."""
    index: dict[str, str] = {}
    for name, (_category, aliases) in _RAW.items():
        index[name.lower()] = name
        for alias in aliases:
            index.setdefault(alias.lower(), name)
    return index


def canonicalize(value: str) -> str:
    return alias_index().get(value.strip().lower(), value.strip())


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def find_skills(
    text: str, *, skills_section: str | None = None
) -> list[SkillHit]:
    """Detect known skills in ``text``.

    ``skills_section`` is a second, narrower view of the same document; hits
    found there are considered high confidence.
    """
    if not text:
        return []
    section = skills_section or ""
    hits: list[SkillHit] = []

    for name, category, patterns in _compiled():
        occurrences = 0
        for pattern in patterns:
            occurrences += sum(1 for _ in pattern.finditer(text))
        if not occurrences:
            continue
        in_section = any(pattern.search(section) for pattern in patterns)
        confidence = 1.0 if in_section else (0.9 if occurrences > 1 else 0.75)
        hits.append(
            SkillHit(
                name=name,
                category=category,
                occurrences=min(occurrences, 20),
                in_skills_section=in_section,
                confidence=confidence,
            )
        )

    hits.sort(key=lambda hit: (_SKILL_ORDER.get(hit.name, 999), hit.name))
    return hits


_SPLIT_SKILLS = re.compile(r"[,;|•·\n\t]|(?:\s+/\s+)|(?:\s{3,})")
_SKILL_NOISE = {
    "skills", "technical skills", "core competencies", "competencies",
    "technologies", "tools", "frameworks", "languages", "databases",
    "platforms", "other", "misc", "miscellaneous", "and", "etc", "various",
    "experience", "professional", "expert", "advanced", "intermediate",
    "beginner", "proficient", "familiar", "working knowledge", "years",
    "responsibilities", "achievements", "summary", "profile", "education",
    "certifications", "projects", "interests", "references",
}


def extract_unknown_skills(
    skills_section: str, known: Iterable[str] = ()
) -> list[str]:
    """Harvest skills the taxonomy does not know (e.g. niche tools).

    Only runs against the ``Skills`` section so it cannot pick up prose.
    """
    if not skills_section:
        return []
    known_lower = {name.lower() for name in known}
    known_lower |= set(alias_index())
    found: list[str] = []

    for line in skills_section.splitlines():
        line = re.sub(r"^[^:]{2,40}:\s*", "", line.strip())  # drop "Languages:"
        for token in _SPLIT_SKILLS.split(line):
            token = token.strip(" .-–—•·()[]{}")
            token = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", token).strip()
            if not token or len(token) < 2 or len(token) > 40:
                continue
            words = token.split()
            if not 1 <= len(words) <= 4:
                continue
            if not re.search(r"[A-Za-z]", token) or not re.search(r"[A-Za-z]{2}", token):
                continue
            if token.lower() in _SKILL_NOISE or token.lower() in known_lower:
                continue
            if re.fullmatch(r"[\d\s.%+-]+", token):
                continue
            if re.search(r"\b(?:and|with|using|for|the|of|to)\b", token.lower()):
                continue
            found.append(token)

    return dedupe_case_insensitive(found)[:30]


def dedupe_case_insensitive(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Job-description weighting
# ---------------------------------------------------------------------------

_REQUIREMENT_CUES = re.compile(
    r"\b(required|requirements?|must[-\s]?have|must|essential|mandatory|"
    r"proficien\w*|strong|expert\w*|hands[-\s]on|experience with|knowledge of)\b",
    re.IGNORECASE,
)
_NICE_TO_HAVE_CUES = re.compile(
    r"\b(nice[-\s]to[-\s]have|preferred|plus|bonus|desirable|advantageous|optional)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WeightedSkill:
    name: str
    category: str
    weight: float
    required: bool
    occurrences: int


def extract_jd_skills(text: str) -> list[WeightedSkill]:
    """Detect JD skills and weight them by frequency and requirement cues."""
    if not text:
        return []
    hits = find_skills(text)
    weighted: list[WeightedSkill] = []

    for hit in hits:
        weight = 1.0 + 0.35 * (min(hit.occurrences, 4) - 1)
        required = False
        nice_to_have = False
        for pattern in _compiled_patterns_for(hit.name):
            for match in pattern.finditer(text):
                window, position = _sentence_window(text, match.start(), match.end())
                requirement_distance = _closest_cue(
                    _REQUIREMENT_CUES, window, position
                )
                nice_distance = _closest_cue(_NICE_TO_HAVE_CUES, window, position)
                # Whichever cue sits closest to the skill decides its status;
                # a "nice to have" elsewhere in the sentence must not downgrade
                # a skill the sentence also marks as required.
                if nice_distance is not None and (
                    requirement_distance is None or nice_distance < requirement_distance
                ):
                    nice_to_have = True
                elif requirement_distance is not None:
                    required = True
        if required:
            weight += 0.9
        elif nice_to_have:
            weight *= 0.65
        weighted.append(
            WeightedSkill(
                name=hit.name,
                category=hit.category,
                weight=round(min(weight, 3.0), 2),
                required=required,
                occurrences=hit.occurrences,
            )
        )

    weighted.sort(key=lambda item: (-item.weight, item.name))
    return weighted


def _compiled_patterns_for(name: str) -> tuple[re.Pattern[str], ...]:
    for canonical, _category, patterns in _compiled():
        if canonical == name:
            return patterns
    return ()


def _closest_cue(
    pattern: re.Pattern[str], window: str, position: int
) -> int | None:
    """Distance from ``position`` to the nearest cue match, if any."""
    best: int | None = None
    for match in pattern.finditer(window):
        distance = abs(match.start() - position)
        if best is None or distance < best:
            best = distance
    return best


def _sentence_window(text: str, start: int, end: int) -> tuple[str, int]:
    """Return the sentence around ``text[start:end]`` and the skill's offset.

    Requirement cues are scoped per sentence: a JD that says "Must have Python
    and Kubernetes. Scala is a nice to have." must not downgrade Kubernetes
    just because a "nice to have" cue appears later in the paragraph.
    """
    left = max(text.rfind(char, 0, start) for char in ".!?\n")
    rights = [index for char in ".!?\n" if (index := text.find(char, end)) != -1]
    right = min(rights) if rights else len(text)
    window_start = left + 1
    window_end = min(len(text), right + 1)
    return text[window_start:window_end], start - window_start


def skill_names(skills: Iterable[object]) -> list[str]:
    """Normalise a mixed list of skill strings / dicts / SkillHit objects."""
    names: list[str] = []
    for skill in skills:
        if isinstance(skill, str):
            names.append(skill)
        elif isinstance(skill, dict):
            value = skill.get("name")
            if value:
                names.append(str(value))
        elif hasattr(skill, "name"):
            names.append(str(skill.name))
    return dedupe_case_insensitive(names)
