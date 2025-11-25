import csv
import random

products = ["t-shirt", "hoodie", "jeans", "jacket", "sneakers", "bag", "cap", "dress", "skirt", "wallet"]
adjectives = ["awesome", "great", "cool", "nice", "beautiful", "amazing", "lovely", "perfect", "bad", "terrible", "poor"]
cities = ["Cairo", "Alexandria", "Giza", "Luxor", "Aswan", "Hurghada", "Sharm El Sheikh", "Mansoura"]

templates = {
    "purchase_intent": [
        "I want to buy a {product}",
        "Do you have this {product} in stock?",
        "How much is the {product}?",
        "Can I order a {product}?",
        "I need a new {product}",
        "Is the {product} available?",
        "Price of {product} please",
        "I'll take two {product}s",
        "Where can I buy this {product}?",
        "Shut up and take my money for that {product}"
    ],
    "product_details": [
        "What material is the {product} made of?",
        "Is the {product} cotton?",
        "What are the dimensions of the {product}?",
        "Does the {product} come in red?",
        "Size chart for {product}?",
        "Is the {product} waterproof?",
        "How do I wash the {product}?",
        "Is the {product} leather?",
        "Does the {product} fit true to size?",
        "Can you show me more photos of the {product}?"
    ],
    "general_praise": [
        "I love your {product}s",
        "This brand is {adjective}",
        "The {product} looks {adjective}",
        "Great quality {product}",
        "Best {product} I ever bought",
        "You guys are {adjective}",
        "Absolutely {adjective} designs",
        "Highly recommend this shop",
        "My favorite local brand",
        "Keep up the {adjective} work"
    ],
    "shipping_inquiry": [
        "Do you ship to {city}?",
        "How long does shipping to {city} take?",
        "Is shipping free to {city}?",
        "When will my order arrive in {city}?",
        "What is the shipping cost to {city}?",
        "Can I track my order to {city}?",
        "Do you deliver to {city}?",
        "I need this delivered to {city} by Friday",
        "Shipping time for {city}?",
        "Is cash on delivery available in {city}?"
    ],
    "complaint": [
        "My {product} arrived damaged",
        "The {product} is poor quality",
        "I am not happy with the {product}",
        "Delivery is too slow",
        "The {product} size is wrong",
        "Customer service is {adjective}",
        "I want to return this {product}",
        "This is too expensive",
        "The {product} looks different than the picture",
        "Worst shopping experience"
    ]
}

def generate_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        intent = random.choice(list(templates.keys()))
        template = random.choice(templates[intent])
        
        text = template.format(
            product=random.choice(products),
            adjective=random.choice(adjectives),
            city=random.choice(cities)
        )
        data.append([text, intent])
        
    return data

if __name__ == "__main__":
    print("Generating synthetic data...")
    data = generate_data(600)
    
    filename = "customer_data.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(data)
        
    print(f"Generated {len(data)} samples in {filename}")
