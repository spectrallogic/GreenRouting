"""Synthetic training data generator for the query classifier.

Generates labeled (query, capability_weights, difficulty) examples across all
capability categories including MIXED-capability queries. This is what the
classifier trains on — it learns to predict what a query NEEDS as weighted
capabilities, not which model to use.

Target: 10,000+ diverse training examples covering:
- Single-capability queries at all difficulty levels
- Mixed-capability queries (code+math, reasoning+knowledge, etc.)
- Natural phrasing variations
- Edge cases and ambiguous queries
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from greenrouting.core.taxonomy import Capability


@dataclass
class TrainingExample:
    """A single training example: query → (capability_weights, difficulty)."""

    query: str
    capability_weights: dict[str, float]  # capability → weight (sums to ~1.0)
    difficulty: int  # 1-5
    expected_output_length: str  # "short", "medium", "long"


# ─── Single-capability query templates ────────────────────────────────────────

SINGLE_TEMPLATES: dict[Capability, dict[int, list[str]]] = {
    Capability.SIMPLE: {
        1: [
            "What is {a} + {b}?",
            "What color is the sky?",
            "How many legs does a dog have?",
            "What is the capital of {country}?",
            "Is {a} greater than {b}?",
            "What day comes after {day}?",
            "Convert {a} meters to centimeters.",
            "What is the opposite of '{word}'?",
            "Spell the word '{word}'.",
            "What language is spoken in {country}?",
            "Who wrote '{book}'?",
            "What year did World War II end?",
            "How many minutes are in an hour?",
            "What is the boiling point of water in Celsius?",
            "Name a fruit that is red.",
            "Is a whale a fish or a mammal?",
            "What comes after the letter {letter} in the alphabet?",
            "How many sides does a triangle have?",
            "What is the plural of '{word}'?",
            "True or false: The Earth is flat.",
            "What season comes after {season}?",
            "How many hours are in a day?",
            "What is the currency of {country}?",
            "Name the largest planet in our solar system.",
            "Is {a} even or odd?",
        ],
        2: [
            "What is the square root of {square}?",
            "Name three countries in {continent}.",
            "What is the chemical symbol for {element}?",
            "Define the word '{vocab_word}'.",
            "What is {a}% of {b}?",
            "List the primary colors.",
            "What is the largest ocean on Earth?",
            "Who painted the Mona Lisa?",
            "How many continents are there?",
            "What is the speed of light approximately?",
            "Name the first three elements of the periodic table.",
            "What is the difference between weather and climate?",
            "How many bones are in the adult human body?",
            "What is the main ingredient in bread?",
            "Name the four cardinal directions.",
        ],
    },
    Capability.REASONING: {
        2: [
            "If all cats are animals, and all animals breathe, do all cats breathe?",
            "A is taller than B. B is taller than C. Who is the shortest?",
            "If it's raining, the ground is wet. The ground is wet. Is it necessarily raining?",
            "There are 5 apples. You take away 3. How many do YOU have?",
            "If a shirt costs $20 after a 50% discount, what was the original price?",
            "Is this statement true or false: 'This sentence is false'?",
            "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
            "I have two coins totaling 30 cents. One is not a nickel. What are they?",
            "Which weighs more, a pound of feathers or a pound of steel?",
            "If you overtake the person in second place, what place are you in?",
        ],
        3: [
            "A farmer has {a} chickens and {b} cows. How many legs are there in total?",
            "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets?",
            "Explain why this argument is flawed: 'All dogs have four legs. My cat has four legs. Therefore, my cat is a dog.'",
            "Three switches outside a room control three bulbs inside. You can enter the room once. How do you figure out which switch controls which bulb?",
            "A snail climbs 3 feet up a wall during the day and slides back 2 feet at night. How many days to reach the top of a {a}-foot wall?",
            "You're in a room with two doors. One leads to freedom, one to death. Two guards — one always lies, one always tells truth. One question. What do you ask?",
            "If it takes 8 people 10 hours to build a wall, how long would it take 4 people?",
            "What's wrong with this logic: 'I've never seen it rain on a Tuesday, so it never rains on Tuesdays'?",
            "A lily pad doubles in size every day. It takes 48 days to cover a lake. On what day does it cover half the lake?",
            "You have 8 identical-looking balls. One is slightly heavier. Using a balance scale twice, find the heavy ball.",
        ],
        4: [
            "A train leaves city A at 60mph. Another leaves city B (300 miles away) at 40mph toward A. When and where do they meet?",
            "You have 12 balls, one is heavier. Using a balance scale only 3 times, find the heavy ball.",
            "Analyze the logical structure of this argument and identify all fallacies: 'Everyone who works hard succeeds. John didn't succeed. Therefore, John didn't work hard. And since lazy people don't work hard, John must be lazy.'",
            "Design an algorithm to determine if a Sudoku puzzle has a unique solution. Explain your reasoning.",
            "A king wants to test 1000 bottles of wine for poison using prisoners. The poison takes exactly 24 hours to kill. He has 10 prisoners and 24 hours. How does he find the poisoned bottle?",
            "Five pirates must divide 100 gold coins. They vote on proposals — if majority rejects, the proposer dies. What does the first pirate propose?",
            "You are in a pitch-dark room with a drawer of socks: 10 black, 10 white, 10 blue. How many must you take to guarantee a matching pair?",
            "There are 100 lockers and 100 students. Student 1 opens all. Student 2 toggles every 2nd. Student 3 every 3rd, etc. Which lockers are open at the end?",
        ],
        5: [
            "Prove that there are infinitely many prime numbers.",
            "Analyze the implications of Gödel's incompleteness theorems on artificial intelligence.",
            "Construct a formal logical proof for: If P→Q and Q→R, then P→R, using natural deduction.",
            "Evaluate the philosophical validity of the Chinese Room argument against strong AI.",
            "Analyze whether P=NP or P≠NP is more likely and present arguments for both sides.",
            "Prove the halting problem is undecidable using diagonalization.",
            "Explain Arrow's impossibility theorem and its implications for voting systems.",
            "Analyze the Ship of Theseus paradox from mereological, four-dimensionalist, and psychological continuity perspectives.",
        ],
    },
    Capability.MATH: {
        1: [
            "What is {a} × {b}?",
            "What is {a} divided by {b}?",
            "Calculate {a} - {b}.",
            "What is {a}²?",
            "Round {decimal} to the nearest integer.",
            "What is {a} + {b} + {c}?",
            "What is half of {a}?",
            "How much is {a} dozen?",
            "What fraction is equivalent to 50%?",
            "What is {a} × 0?",
        ],
        2: [
            "Solve for x: {a}x + {b} = {c}",
            "What is the area of a triangle with base {a} and height {b}?",
            "Calculate the average of {a}, {b}, {c}, and {d}.",
            "Convert {fraction} to a decimal.",
            "What is {a}! (factorial)?",
            "Find the perimeter of a rectangle with length {a} and width {b}.",
            "What is the GCD of {a} and {b}?",
            "Simplify the fraction {a}/{big_a}.",
            "What is 2^{small_a}?",
            "How many combinations of 2 items from {small_a} items?",
        ],
        3: [
            "Solve the quadratic equation: x² + {b}x + {c} = 0",
            "Find the derivative of f(x) = {a}x³ + {b}x² + {c}x",
            "Calculate the probability of rolling a sum of {target} with two dice.",
            "Find the area under the curve y = x² from x = 0 to x = {small_a}.",
            "A population grows at {rate}% per year. Starting from {start}, what is it after {years} years?",
            "Find the eigenvalues of the 2x2 matrix [[{small_a}, {b}], [{c}, {d}]].",
            "What is the Taylor series expansion of e^x around x=0, first 5 terms?",
            "Solve the system: 2x + 3y = {a}, 4x - y = {b}",
            "Find the standard deviation of the dataset: {a}, {b}, {c}, {d}, {small_a}.",
            "Calculate the determinant of [[{small_a}, {b}], [{c}, {d}]].",
            "What is the integral of sin(x)cos(x) dx?",
            "Find the limit as x→0 of sin(x)/x.",
        ],
        4: [
            "Evaluate the integral ∫(0 to ∞) e^(-x²) dx",
            "Solve the system of differential equations: dx/dt = {a}x + {b}y, dy/dt = {c}x + {d}y",
            "Prove that √2 is irrational.",
            "Find the eigenvalues of the matrix [[{small_a}, {b}, 0], [0, {c}, {d}], [{small_a}, 0, {b}]].",
            "Derive the formula for the sum of the first n terms of a geometric series.",
            "Compute the Fourier transform of a rectangular pulse function.",
            "Solve the recurrence relation: T(n) = 2T(n/2) + n, T(1) = 1.",
            "Prove by induction that the sum of first n odd numbers equals n².",
            "Find all solutions to z⁴ = -16 in the complex plane.",
        ],
        5: [
            "Prove the Cauchy-Schwarz inequality for inner product spaces.",
            "Solve the partial differential equation ∂²u/∂t² = c²∂²u/∂x² with given boundary conditions.",
            "Prove that every bounded sequence in ℝⁿ has a convergent subsequence.",
            "Derive the Black-Scholes equation from first principles.",
            "Prove the Fundamental Theorem of Algebra using Liouville's theorem.",
            "Prove the Central Limit Theorem for i.i.d. random variables.",
            "Solve the Navier-Stokes equations for laminar flow between parallel plates.",
            "Derive the Euler-Lagrange equation from the calculus of variations.",
        ],
    },
    Capability.CODE: {
        1: [
            "Write a Python function that returns 'Hello, World!'",
            "Write a function to add two numbers in Python.",
            "How do you print a list in Python?",
            "Write a for loop that prints numbers 1 to {a}.",
            "How do you create a dictionary in Python?",
            "Write a function that returns True if a number is positive.",
            "How do you read a file in Python?",
            "Write code to create a list of numbers from 1 to {a}.",
            "What does len() do in Python?",
            "How do you concatenate two strings in Python?",
        ],
        2: [
            "Write a function to check if a string is a palindrome.",
            "Write a function to find the maximum element in a list.",
            "Implement FizzBuzz in Python.",
            "Write a function to reverse a string without using built-in reverse.",
            "Write a function that counts the occurrences of each character in a string.",
            "Write a function to check if two strings are anagrams.",
            "Implement a simple calculator that handles +, -, *, /.",
            "Write code to remove duplicates from a list while preserving order.",
            "Write a function to flatten a nested list.",
            "Implement a function that generates the Fibonacci sequence up to n terms.",
            "Write a function to check if a number is prime.",
            "Write code to sort a dictionary by its values.",
        ],
        3: [
            "Implement a binary search algorithm in Python.",
            "Write a function to find all permutations of a string.",
            "Implement a stack using a linked list.",
            "Write a function to detect a cycle in a linked list.",
            "Implement merge sort and explain its time complexity.",
            "Write a REST API endpoint using FastAPI that handles CRUD operations for a user model.",
            "Implement a basic hash table with collision handling.",
            "Write a decorator that caches function results (memoization).",
            "Implement BFS and DFS for a graph represented as an adjacency list.",
            "Write a function to find the longest common subsequence of two strings.",
            "Implement a priority queue using a heap.",
            "Write a context manager for database connections.",
            "Implement the observer pattern in Python.",
        ],
        4: [
            "Implement an LRU cache with O(1) get and put operations.",
            "Write a function to serialize and deserialize a binary tree.",
            "Implement a thread-safe producer-consumer queue.",
            "Design and implement a rate limiter using the token bucket algorithm.",
            "Write a coroutine-based async web scraper with retry logic and rate limiting.",
            "Implement a Trie with insert, search, and prefix matching.",
            "Write a SQL query optimizer that can reorder joins based on estimated cardinality.",
            "Implement a basic event-driven architecture with an event bus.",
            "Write a custom connection pool for database connections with timeout handling.",
            "Implement the Aho-Corasick algorithm for multi-pattern string matching.",
        ],
        5: [
            "Implement a B-tree with insert, delete, and search operations.",
            "Design and implement a distributed consensus algorithm (simplified Raft).",
            "Implement a garbage collector using mark-and-sweep algorithm.",
            "Build a simple bytecode interpreter for a stack-based virtual machine.",
            "Implement a lock-free concurrent hash map.",
            "Write a compiler front-end that lexes, parses, and type-checks a simple language.",
            "Implement a persistent (immutable) red-black tree.",
            "Design a distributed key-value store with consistent hashing and replication.",
        ],
    },
    Capability.KNOWLEDGE: {
        1: [
            "What is photosynthesis?",
            "Who was the first president of the United States?",
            "What is the chemical formula for water?",
            "What continent is Brazil in?",
            "What is DNA?",
            "What are the states of matter?",
            "Who discovered gravity?",
            "What is the largest organ in the human body?",
            "What do plants need to grow?",
            "What is the closest star to Earth?",
        ],
        2: [
            "Explain how vaccines work.",
            "What caused the fall of the Roman Empire?",
            "How does a combustion engine work?",
            "What is the difference between mitosis and meiosis?",
            "Explain the water cycle.",
            "What is the greenhouse effect?",
            "How does the internet work?",
            "What is the difference between a virus and a bacteria?",
            "Explain supply and demand in economics.",
            "How do airplanes fly?",
            "What is evolution by natural selection?",
            "What is the electromagnetic spectrum?",
        ],
        3: [
            "Explain quantum entanglement in simple terms.",
            "What are the key differences between TCP and UDP?",
            "Describe the process of CRISPR gene editing.",
            "Explain how transformer models work in machine learning.",
            "What is the significance of the Higgs boson discovery?",
            "Explain how blockchain technology works.",
            "What is the difference between machine learning and deep learning?",
            "Describe how nuclear fission reactors generate electricity.",
            "Explain the concept of entropy in thermodynamics.",
            "How does GPS determine location?",
            "What is CRISPR and how does it edit genes?",
            "Explain the difference between supervised and unsupervised learning.",
        ],
        4: [
            "Explain the mechanism of action of mRNA vaccines at the molecular level.",
            "Describe the theoretical framework behind quantum computing and its advantages over classical computing.",
            "Analyze the economic factors that led to the 2008 financial crisis.",
            "Explain the Standard Model of particle physics and its limitations.",
            "Describe how compiler optimization works, including SSA form and register allocation.",
            "Explain the CAP theorem and its implications for distributed systems.",
            "Describe the mechanism of CRISPR-Cas9 at the molecular level including PAM recognition and DSB repair.",
            "Explain how large language models are trained, including pretraining, RLHF, and alignment.",
            "Describe the neuroscience of memory formation and retrieval.",
        ],
        5: [
            "Provide a detailed analysis of the renormalization group in quantum field theory.",
            "Explain the mathematical foundations of general relativity including the Einstein field equations.",
            "Analyze the implications of the holographic principle for our understanding of quantum gravity.",
            "Describe the current state of research on the Yang-Mills existence and mass gap problem.",
            "Explain the proof strategy behind the recent advances in the Langlands program.",
            "Describe the AdS/CFT correspondence and its implications for quantum gravity.",
            "Explain topological quantum computing and how non-abelian anyons could be used for fault-tolerant computation.",
        ],
    },
    Capability.CREATIVE: {
        1: [
            "Write a haiku about {topic}.",
            "Come up with a name for a pet {animal}.",
            "Write a one-sentence story about {topic}.",
            "Suggest a title for a book about {topic}.",
            "Write a fun fact about {topic}.",
            "Come up with a catchy slogan for {topic}.",
        ],
        2: [
            "Write a short poem about {topic}.",
            "Create a metaphor for {concept}.",
            "Write a product description for {product}.",
            "Come up with 5 creative names for a {business_type} business.",
            "Write a short joke about {topic}.",
            "Write a limerick about {topic}.",
            "Create a simile comparing {topic} to something unexpected.",
            "Write a 50-word flash fiction about {topic}.",
            "Come up with a creative insult a pirate would say.",
            "Write a fortune cookie message about {concept}.",
        ],
        3: [
            "Write a short story (200 words) about {topic} with a twist ending.",
            "Create a detailed character description for a {genre} novel protagonist.",
            "Write a persuasive essay outline about {topic}.",
            "Compose a sonnet about {topic} following proper meter and rhyme scheme.",
            "Write a dialogue between two characters debating {topic}.",
            "Create a backstory for a villain who believes they are saving the world.",
            "Write a monologue from the perspective of an inanimate object.",
            "Write a {genre} story opening that hooks the reader in the first paragraph.",
            "Create a fictional Wikipedia entry for an invented historical event.",
            "Write a series of diary entries from an astronaut's first week on Mars.",
        ],
        4: [
            "Write a 500-word short story in the style of {author} about {topic}.",
            "Create a detailed world-building document for a {genre} setting.",
            "Write a compelling cover letter for a {job_title} position that stands out.",
            "Compose a multi-stanza ballad telling the story of {event}.",
            "Write a satirical news article about {topic} in the style of The Onion.",
            "Create a complete magic system with rules, costs, and limitations for a fantasy novel.",
            "Write a monologue for a character who just realized they're in a simulation.",
            "Write a short play (two scenes) exploring the theme of {concept}.",
        ],
        5: [
            "Write the opening chapter of a novel that establishes an unreliable narrator.",
            "Create a complete screenplay for a 10-minute short film about {topic}.",
            "Write a literary analysis essay comparing themes in two {genre} works.",
            "Compose a series of interconnected poems that tell a complete narrative arc.",
            "Write a philosophical dialogue in the style of Plato exploring {concept}.",
            "Write a multi-perspective short story where three narrators describe the same event differently.",
            "Create an epistolary short story told through emails, texts, and social media posts.",
        ],
    },
    Capability.INSTRUCTION: {
        2: [
            "List exactly 3 benefits of exercise. Number them 1-3.",
            "Explain {topic} in exactly two sentences.",
            "Write your response in all lowercase letters: What is machine learning?",
            "Respond with only 'yes' or 'no': Is Python an interpreted language?",
            "Give me 5 items, each on a new line, no numbering.",
            "Answer in exactly 10 words: Why is the sky blue?",
            "List pros and cons of {topic} in a two-column format.",
        ],
        3: [
            "Write a recipe for {dish}. Format it with 'Ingredients:' and 'Steps:' sections. Each step should be numbered.",
            "Explain {topic} to three audiences: a child, a teenager, and an expert. Label each section.",
            "Summarize this concept in exactly {n} bullet points: {topic}",
            "Write a response that contains exactly {n} paragraphs, each starting with a different letter of the alphabet.",
            "Create a FAQ section with exactly 5 questions and answers about {topic}.",
            "Write a comparison table with 5 rows comparing {topic} and {concept}.",
            "Give me a step-by-step guide with exactly 7 steps for {topic}.",
        ],
        4: [
            "Write a technical blog post about {topic}. Include: an introduction, exactly 3 main sections with headers, code examples in each section, and a conclusion. Keep it under 1000 words.",
            "Create a structured comparison of {topic} vs {concept}. Use a table format with at least 5 comparison criteria. End with a recommendation.",
            "Write instructions for {task} that follow these constraints: use only imperative sentences, include exactly 10 steps, and add a warning box after step 5.",
            "Write a tutorial that alternates between explanation paragraphs and code blocks. Must have exactly 4 of each.",
        ],
        5: [
            "Write a technical RFC document proposing a solution for {topic}. Follow the standard RFC format with: Abstract, Introduction, Motivation, Specification, Security Considerations, and References sections.",
            "Create a complete lesson plan for teaching {topic} that includes: learning objectives (using Bloom's taxonomy), activities for each objective, assessment rubric, and differentiation strategies.",
        ],
    },
    Capability.MULTILINGUAL: {
        1: [
            "Translate 'hello' to {language}.",
            "How do you say 'thank you' in {language}?",
            "What does '{foreign_word}' mean in English?",
            "How do you say 'goodbye' in {language}?",
            "Count from 1 to 5 in {language}.",
        ],
        2: [
            "Translate this sentence to {language}: '{sentence}'",
            "Write a greeting in {language} and explain the cultural context.",
            "What are the basic counting words (1-10) in {language}?",
            "How do you order food at a restaurant in {language}?",
            "Write 'I love programming' in {language}.",
        ],
        3: [
            "Translate this paragraph to {language}, preserving the tone and idioms: '{paragraph}'",
            "Write a short business email in {language} requesting a meeting.",
            "Explain the grammatical differences between {language_a} and {language_b}.",
            "Translate these technical terms to {language}: API, machine learning, database.",
            "Write a formal apology letter in {language}.",
        ],
        4: [
            "Translate this technical document from English to {language}, preserving all technical terminology and formatting.",
            "Write a culturally appropriate marketing slogan in {language} for {product}.",
            "Analyze the nuances lost in translating '{foreign_word}' to English.",
            "Write a legal disclaimer in {language} for a software product.",
        ],
        5: [
            "Translate this poem preserving meter, rhyme scheme, and cultural references into {language}.",
            "Write a legal contract clause in {language} following local legal conventions.",
            "Analyze how the concept of '{concept}' differs across {language_a}, {language_b}, and English-speaking cultures.",
        ],
    },
}

# ─── Mixed-capability query templates ─────────────────────────────────────────
# These are the KEY addition — real queries often need multiple capabilities

MIXED_TEMPLATES: list[tuple[dict[str, float], int, str, list[str]]] = [
    # (capability_weights, difficulty, output_length, [templates])

    # Code + Math
    ({"code": 0.5, "math": 0.35, "reasoning": 0.15}, 3, "long", [
        "Write a Python function to solve a system of linear equations using Gaussian elimination.",
        "Implement Newton's method for finding roots of a polynomial in Python.",
        "Write code to compute the Monte Carlo estimate of pi.",
        "Implement matrix multiplication from scratch in Python, then use it to solve Ax = b.",
        "Write a Python script that numerically integrates a function using Simpson's rule.",
        "Implement the Fast Fourier Transform (FFT) algorithm in Python.",
        "Write code to fit a polynomial curve to a set of data points using least squares.",
        "Implement a function to compute the PageRank of nodes in a graph.",
    ]),
    ({"code": 0.5, "math": 0.35, "reasoning": 0.15}, 4, "long", [
        "Write a Python script that solves this differential equation: dy/dx = y*sin(x), y(0) = 1, using RK4.",
        "Implement a neural network from scratch (no frameworks) that learns XOR.",
        "Write an optimized matrix library supporting addition, multiplication, inversion, and eigenvalue decomposition.",
        "Implement gradient descent with backpropagation for a simple 2-layer neural network.",
        "Write a Black-Scholes option pricing calculator with Greeks computation.",
    ]),

    # Code + Knowledge
    ({"code": 0.55, "knowledge": 0.3, "reasoning": 0.15}, 3, "long", [
        "Write a Python script that fetches weather data from an API and plots temperature trends.",
        "Implement a basic search engine: crawl pages, build an inverted index, rank by TF-IDF.",
        "Write code to parse and analyze a CSV of stock market data, computing moving averages and RSI.",
        "Build a simple chatbot using regex patterns and a knowledge base dictionary.",
        "Write a Python script that scrapes a website and stores the data in SQLite.",
        "Implement a recommendation system using collaborative filtering.",
    ]),
    ({"code": 0.45, "knowledge": 0.35, "reasoning": 0.2}, 4, "long", [
        "Implement a simple version of the MapReduce paradigm in Python and explain the distributed computing concepts.",
        "Write a blockchain implementation from scratch. Explain the cryptographic principles behind each component.",
        "Build a basic compiler for arithmetic expressions: lexer, parser, AST, and code generation. Explain each phase.",
        "Implement a basic version of the Raft consensus algorithm and explain the distributed systems concepts.",
    ]),

    # Reasoning + Knowledge
    ({"reasoning": 0.5, "knowledge": 0.4, "instruction": 0.1}, 3, "long", [
        "Compare and contrast capitalism and socialism. Analyze the strengths and weaknesses of each using historical examples.",
        "Explain why antibiotics don't work on viruses. Walk through the biological reasoning step by step.",
        "Analyze the trolley problem from utilitarian, deontological, and virtue ethics perspectives.",
        "Why did the dinosaurs go extinct? Evaluate the competing theories and evidence for each.",
        "Explain the paradox of tolerance and analyze its implications for free speech policy.",
        "Analyze the pros and cons of nuclear energy versus renewable energy sources.",
    ]),
    ({"reasoning": 0.55, "knowledge": 0.35, "creative": 0.1}, 4, "long", [
        "Analyze the ethical implications of AI-generated art. Consider perspectives of artists, consumers, and AI developers.",
        "Evaluate the argument that consciousness is an emergent property of complex systems. Use evidence from neuroscience and philosophy.",
        "Analyze whether social media has been net positive or negative for democracy. Use specific examples from multiple countries.",
        "Compare the Copenhagen and Many-Worlds interpretations of quantum mechanics. Which has stronger philosophical support?",
    ]),

    # Creative + Knowledge
    ({"creative": 0.55, "knowledge": 0.3, "instruction": 0.15}, 3, "long", [
        "Write a short story set during the French Revolution that's historically accurate.",
        "Create a science fiction story that accurately incorporates the concept of time dilation.",
        "Write a dialogue between Einstein and Newton discussing modern physics.",
        "Create a children's story that teaches the water cycle in an engaging way.",
        "Write a historical fiction scene set on the Titanic, accurate to the timeline of events.",
    ]),
    ({"creative": 0.5, "knowledge": 0.3, "reasoning": 0.2}, 4, "long", [
        "Write a hard science fiction short story that accurately portrays the challenges of interstellar travel, including relativistic effects.",
        "Create a detective story where the clues involve real chemistry and the reader could theoretically solve it.",
        "Write a philosophical dialogue between a human and an AI about consciousness, incorporating real AI research.",
    ]),

    # Math + Reasoning
    ({"math": 0.5, "reasoning": 0.4, "knowledge": 0.1}, 4, "long", [
        "Prove that the set of rational numbers is countable but the set of real numbers is not.",
        "Explain the Monty Hall problem, prove the correct solution mathematically, and explain why human intuition fails.",
        "Derive Bayes' theorem from first principles and explain three real-world applications with worked examples.",
        "Analyze the mathematics behind RSA encryption. Why does it work? What would break it?",
    ]),
    ({"math": 0.45, "reasoning": 0.4, "knowledge": 0.15}, 5, "long", [
        "Prove the impossibility of trisecting an arbitrary angle with compass and straightedge using field theory.",
        "Explain why the continuum hypothesis is independent of ZFC. What are the implications?",
        "Derive the Euler-Lagrange equation and use it to solve the brachistochrone problem.",
    ]),

    # Code + Creative
    ({"code": 0.5, "creative": 0.35, "reasoning": 0.15}, 3, "long", [
        "Write a Python program that generates ASCII art from text input.",
        "Create a text-based adventure game in Python with at least 5 rooms and items.",
        "Write a Python script that generates random poetry using Markov chains.",
        "Build a terminal-based snake game in Python.",
        "Write a program that creates generative art using turtle graphics.",
    ]),

    # Multilingual + Knowledge
    ({"multilingual": 0.5, "knowledge": 0.35, "instruction": 0.15}, 3, "medium", [
        "Explain the concept of 'wabi-sabi' in Japanese culture and provide examples of how it appears in daily life. Include key Japanese terms.",
        "Describe the differences between formal and informal speech in Korean, with examples of each register.",
        "Explain the Arabic root system (trilateral roots) and show how it creates related words from a single root.",
        "Describe the tonal system in Mandarin Chinese and explain how tone changes meaning, with examples.",
    ]),

    # Instruction + Code
    ({"instruction": 0.4, "code": 0.45, "knowledge": 0.15}, 3, "long", [
        "Write a step-by-step tutorial for building a REST API in Flask. Include exactly 6 steps, each with a code block and explanation.",
        "Create a beginner's guide to Git with exactly 10 commands, each explained with an example. Format as a numbered list.",
        "Write documentation for a Python class with exactly 5 methods. Include docstrings, type hints, and usage examples for each.",
    ]),
    ({"instruction": 0.35, "code": 0.5, "reasoning": 0.15}, 4, "long", [
        "Write a complete CI/CD pipeline configuration (GitHub Actions) for a Python project. Include: linting, testing, building, and deploying. Add comments explaining each step.",
        "Create a comprehensive error handling guide for a REST API. Include: error types, status codes, error response format, middleware implementation, and testing strategy.",
    ]),

    # Simple + Instruction (formatted simple queries)
    ({"simple": 0.6, "instruction": 0.4}, 1, "short", [
        "In one word, what color is a banana?",
        "Answer with just a number: How many continents are there?",
        "Yes or no: Is the sun a star?",
        "In exactly 3 words, describe the weather today.",
        "Name one planet. Just the name, nothing else.",
    ]),

    # Reasoning + Code
    ({"reasoning": 0.45, "code": 0.4, "math": 0.15}, 4, "long", [
        "Design and implement a solution to the N-Queens problem. Explain your algorithm choice and prove its correctness.",
        "Implement A* pathfinding. Prove that it's optimal when the heuristic is admissible and consistent.",
        "Write a SAT solver using DPLL algorithm. Explain the logical foundations of boolean satisfiability.",
        "Design a type inference algorithm for a simple functional language. Explain the unification process.",
    ]),

    # Knowledge + Multilingual
    ({"knowledge": 0.45, "multilingual": 0.4, "creative": 0.15}, 3, "long", [
        "Explain the history and cultural significance of haiku poetry. Include examples in Japanese with English translations.",
        "Describe the concept of 'hygge' in Danish culture. How does it compare to similar concepts in other cultures?",
        "Explain the origin and evolution of the word 'algorithm' from Arabic to modern usage across languages.",
    ]),

    # Three-way even split
    ({"code": 0.35, "math": 0.35, "reasoning": 0.3}, 5, "long", [
        "Implement a cryptographic protocol (Diffie-Hellman key exchange) from scratch. Prove its security properties mathematically.",
        "Build a neural network that learns to play tic-tac-toe using Q-learning. Derive the update equations and prove convergence.",
        "Implement a numerical PDE solver using the finite element method. Derive the weak formulation and prove stability.",
    ]),
]


# ─── Fill-in values ──────────────────────────────────────────────────────────

FILL_VALUES: dict[str, list] = {
    "a": list(range(2, 100)),
    "b": list(range(2, 50)),
    "c": list(range(1, 30)),
    "d": list(range(1, 20)),
    "small_a": list(range(2, 12)),
    "big_a": list(range(100, 1000, 50)),
    "country": [
        "France", "Japan", "Brazil", "Egypt", "India", "Canada", "Australia",
        "Mexico", "Germany", "Kenya", "South Korea", "Italy", "Spain", "China",
        "Argentina", "Thailand", "Nigeria", "Sweden", "Turkey", "Russia",
    ],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "letter": ["A", "B", "C", "D", "F", "G", "H", "K", "M", "P", "R", "S", "T", "W"],
    "season": ["spring", "summer", "fall", "winter"],
    "word": [
        "happy", "cold", "fast", "big", "light", "ancient", "difficult", "beautiful",
        "gentle", "fierce", "quiet", "bright", "dark", "smooth", "rough",
        "child", "mouse", "leaf", "woman", "tooth", "cactus", "crisis",
    ],
    "book": [
        "Harry Potter", "1984", "Pride and Prejudice", "The Great Gatsby",
        "To Kill a Mockingbird", "Brave New World", "The Hobbit", "Dune",
        "Fahrenheit 451", "The Catcher in the Rye",
    ],
    "square": [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225],
    "continent": ["Europe", "Asia", "Africa", "South America", "North America", "Oceania"],
    "element": [
        "gold", "iron", "oxygen", "hydrogen", "carbon", "sodium", "helium",
        "nitrogen", "silver", "copper", "zinc", "lead", "mercury", "neon",
    ],
    "vocab_word": [
        "ephemeral", "ubiquitous", "pragmatic", "eloquent", "resilient", "paradigm",
        "serendipity", "enigmatic", "juxtaposition", "quintessential", "mellifluous",
        "sycophantic", "obfuscate", "perfunctory", "cacophony",
    ],
    "fraction": ["1/4", "3/8", "5/6", "7/12", "2/3", "5/16", "11/13", "7/9"],
    "decimal": [3.7, 4.2, 8.9, 2.1, 6.5, 11.3, 0.4, 15.8, 99.6, 7.77],
    "rate": [2, 3, 5, 7, 8, 10, 12, 15],
    "start": [1000, 5000, 10000, 50000, 100000, 1000000],
    "years": [5, 10, 15, 20, 30, 50],
    "target": [5, 6, 7, 8, 9, 10, 11, 12],
    "n": [3, 4, 5, 6, 7, 8, 10],
    "topic": [
        "climate change", "artificial intelligence", "space exploration", "music",
        "cooking", "friendship", "time travel", "the ocean", "technology", "nature",
        "social media", "education", "quantum computing", "renewable energy",
        "mental health", "cryptocurrency", "remote work", "biodiversity",
        "cybersecurity", "urbanization", "aging", "genetic engineering",
    ],
    "concept": [
        "loneliness", "freedom", "time", "innovation", "trust", "chaos",
        "identity", "justice", "memory", "power", "empathy", "silence",
        "consciousness", "entropy", "beauty", "truth",
    ],
    "animal": ["cat", "dog", "hamster", "parrot", "rabbit", "turtle", "goldfish", "ferret"],
    "product": [
        "smart water bottle", "AI-powered notebook", "eco-friendly backpack",
        "solar phone charger", "noise-cancelling sleep mask", "modular furniture kit",
    ],
    "business_type": [
        "coffee shop", "tech startup", "bakery", "fitness studio", "bookstore",
        "food truck", "consulting firm", "art gallery",
    ],
    "genre": ["sci-fi", "fantasy", "mystery", "romance", "horror", "thriller", "dystopian"],
    "author": [
        "Hemingway", "Tolkien", "Austen", "Orwell", "Kafka", "Poe",
        "Asimov", "Bradbury", "Atwood", "Vonnegut",
    ],
    "job_title": [
        "software engineer", "data scientist", "product manager", "designer",
        "marketing director", "CTO", "research scientist",
    ],
    "event": [
        "the moon landing", "a revolution", "a great migration",
        "the last day of school", "a first contact with aliens",
    ],
    "dish": [
        "pasta carbonara", "chicken curry", "chocolate cake", "sushi rolls",
        "pad thai", "beef stew", "tiramisu", "ramen", "tacos",
    ],
    "task": [
        "setting up a home network", "changing a car tire", "deploying a web app",
        "installing a ceiling fan", "setting up a development environment",
    ],
    "language": [
        "Spanish", "French", "Japanese", "Mandarin", "Arabic", "German",
        "Korean", "Portuguese", "Russian", "Hindi", "Italian", "Swahili",
    ],
    "language_a": ["English", "Spanish", "French", "Japanese", "German", "Mandarin"],
    "language_b": ["Spanish", "French", "German", "English", "Korean", "Arabic"],
    "foreign_word": [
        "schadenfreude", "saudade", "ikigai", "hygge", "ubuntu", "wabi-sabi",
        "lagom", "gezellig", "meraki", "toska",
    ],
    "sentence": [
        "The weather is beautiful today.",
        "I would like to order coffee please.",
        "Where is the nearest train station?",
        "Can you recommend a good restaurant?",
        "I'm learning your language and enjoying it.",
        "What time does the museum close?",
    ],
    "paragraph": [
        "Technology is reshaping how we communicate. What once took days now takes seconds. But are we truly more connected?",
        "The city wakes slowly on Sunday mornings. Coffee shops fill with quiet conversation. There's a gentleness to the pace.",
        "Education is not just about memorizing facts. It's about learning to think critically and solve problems creatively.",
    ],
}

# Output length expectations per capability and difficulty
OUTPUT_LENGTHS: dict[Capability, dict[int, str]] = {
    Capability.SIMPLE: {1: "short", 2: "short"},
    Capability.REASONING: {2: "medium", 3: "medium", 4: "long", 5: "long"},
    Capability.MATH: {1: "short", 2: "short", 3: "medium", 4: "long", 5: "long"},
    Capability.CODE: {1: "short", 2: "medium", 3: "long", 4: "long", 5: "long"},
    Capability.KNOWLEDGE: {1: "short", 2: "medium", 3: "medium", 4: "long", 5: "long"},
    Capability.CREATIVE: {1: "short", 2: "medium", 3: "long", 4: "long", 5: "long"},
    Capability.INSTRUCTION: {2: "medium", 3: "long", 4: "long", 5: "long"},
    Capability.MULTILINGUAL: {1: "short", 2: "short", 3: "medium", 4: "long", 5: "long"},
}


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template string with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, str(rng.choice(values)), 1)
    return result


def _add_natural_variations(query: str, rng: random.Random) -> str:
    """Add natural language variations to make queries more diverse."""
    prefixes = [
        "", "", "", "",  # Most queries have no prefix (weighted)
        "Can you ", "Please ", "Help me ", "I need to ",
        "Hey, ", "Quick question: ", "I was wondering, ",
        "Could you help me ", "I'd like you to ",
    ]
    suffixes = [
        "", "", "", "", "",  # Most have no suffix
        " Thanks.", " Please.", " Thanks in advance.",
        " I'm in a hurry.", " This is urgent.",
        " Explain your reasoning.", " Be thorough.",
        " Keep it simple.", " Be brief.",
    ]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)

    # Don't prefix if it would be grammatically weird
    if prefix and query[0].isupper() and prefix.strip():
        query = query[0].lower() + query[1:]

    return prefix + query + suffix


def generate_dataset(
    n_per_category: int = 50,
    seed: int = 42,
) -> list[TrainingExample]:
    """Generate a synthetic training dataset.

    Args:
        n_per_category: Number of examples per (capability, difficulty) pair.
            Default 50 generates ~3,500 examples.
            Use 150+ for 10,000+ examples.
        seed: Random seed for reproducibility.

    Returns:
        List of TrainingExample objects.
    """
    rng = random.Random(seed)
    examples: list[TrainingExample] = []

    # ── Single-capability examples ────────────────────────────────────
    for capability, difficulty_templates in SINGLE_TEMPLATES.items():
        for difficulty, templates in difficulty_templates.items():
            for _ in range(n_per_category):
                template = rng.choice(templates)
                query = _fill_template(template, rng)
                query = _add_natural_variations(query, rng)
                output_length = OUTPUT_LENGTHS.get(capability, {}).get(difficulty, "medium")

                examples.append(TrainingExample(
                    query=query,
                    capability_weights={capability.value: 1.0},
                    difficulty=difficulty,
                    expected_output_length=output_length,
                ))

    # ── Mixed-capability examples ─────────────────────────────────────
    # Generate more mixed examples since they're harder to classify
    mixed_per_template = max(n_per_category // 2, 10)
    for weights, difficulty, output_length, templates in MIXED_TEMPLATES:
        for _ in range(mixed_per_template):
            template = rng.choice(templates)
            query = _fill_template(template, rng)
            query = _add_natural_variations(query, rng)

            examples.append(TrainingExample(
                query=query,
                capability_weights=dict(weights),
                difficulty=difficulty,
                expected_output_length=output_length,
            ))

    rng.shuffle(examples)
    return examples


def save_dataset(examples: list[TrainingExample], path: str | Path) -> None:
    """Save dataset to JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def load_dataset(path: str | Path) -> list[TrainingExample]:
    """Load dataset from JSONL format."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(TrainingExample(**data))
    return examples
