import json
import random
from pathlib import Path
from copy import deepcopy

random.seed(42)

# Load product data
with open("data/products.json", encoding="utf-8") as f:
    products_data = json.load(f)
products = products_data["items"]

# Number to text map
NUM2TEXT = {1: "one", 2: "two", 3: "three"}

# Format quantity as either number or word
def format_quantity(quantity):
    return random.choice([str(quantity), NUM2TEXT[quantity]])

# Generate a basket item with optional options
def generate_item():
    product = random.choice(products)
    quantity = random.randint(1, 3)
    options = []
    if product.get("options"):
        sampled_options = random.sample(product["options"], k=random.randint(0, len(product["options"])))
        for opt in sampled_options:
            options.append({
                "id": opt["id"],
                "quantity": 1,
                "price": opt["price"]
            })
    total_price = product["price"] * quantity + sum(opt["price"] for opt in options)
    return {
        "id": product["id"],
        "quantity": quantity,
        "price": round(total_price, 2),
        "options": options
    }

# Generate a human-like query for previous step
def generate_prev_query(items):
    item_strs = [
        f"{format_quantity(item['quantity'])} {next(p['name'] for p in products if p['id'] == item['id'])}"
        for item in items
    ]
    joined = ", ".join(item_strs)
    return random.choice([
        f"I'd like {joined}",
        f"Can I get {joined} please?",
        f"Add {joined} to my basket",
        f"I'll have {joined}",
        f"Just {joined} for now"
    ])

# Main example generation logic
def generate_example(intent):
    # Simulate memory
    prev_items = [generate_item() for _ in range(random.randint(1, 2))]
    prev_query = generate_prev_query(prev_items)
    prev_response = {
        "intent": "UPDATE_BASKET",
        "basket": deepcopy(prev_items),
        "response": f"Sure, I’ve added your items to the basket."
    }

    messages = [
        {"role": "user", "content": prev_query},
        {"role": "assistant", "content": prev_response["response"]}
    ]

    if intent == "UPDATE_BASKET":
        new_items = [generate_item() for _ in range(random.randint(1, 3))]
        item_strs = [
            f"{format_quantity(item['quantity'])} {next(p['name'] for p in products if p['id'] == item['id'])}"
            for item in new_items
        ]
        instruction = random.choice([
            f"I want {', '.join(item_strs)}",
            f"Please add {', '.join(item_strs)}",
            f"Can I also get {', '.join(item_strs)}?"
        ])
        updated_basket = prev_items + new_items
        assistant_response = f"Sure! I’ve added {', '.join(item_strs)}. Your basket has been updated."

    elif intent == "CLEAR_BASKET":
        instruction = random.choice([
            "Can you clear my order?",
            "Start again please",
            "Remove everything",
            "Delete the basket"
        ])
        updated_basket = []
        assistant_response = "Okay, I’ve cleared your basket."

    elif intent == "GO_BACK_ONE_STEP":
        if len(prev_items) > 1:
            updated_basket = prev_items[:-1]
            removed_item = prev_items[-1]
        else:
            updated_basket = []
            removed_item = prev_items[0]
        item_name = next(p['name'] for p in products if p['id'] == removed_item['id'])
        instruction = random.choice([
            f"Actually, remove the {item_name}",
            "Undo the last item",
            "Go back one step"
        ])
        assistant_response = f"Okay, I’ve removed the {item_name}."

    elif intent == "MAKE_PAYMENT":
        instruction = random.choice([
            "That’s all, I’m ready to pay",
            "Proceed to checkout",
            "Can I pay now?"
        ])
        updated_basket = prev_items
        total = round(sum(i["price"] for i in updated_basket), 2)
        assistant_response = f"Great, your total is £{total}. Please drive to the next window."

    elif intent == "INVALID_REQUEST":
        instruction = random.choice([
            "Do you have any vegan socks?",
            "Can I buy a TV here?",
            "Tell me a joke"
        ])
        updated_basket = prev_items
        assistant_response = "Sorry, I didn’t understand that. Could you rephrase your order?"

    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": assistant_response})

    return {
        "messages": messages,
        "output": {
            "intent": intent,
            "basket": updated_basket,
            "response": assistant_response
        }
    }

# Generate the dataset
examples = []
INTENTS = ["UPDATE_BASKET", "CLEAR_BASKET", "GO_BACK_ONE_STEP", "MAKE_PAYMENT", "INVALID_REQUEST"]
for _ in range(1000):
    intent = random.choice(INTENTS)
    examples.append(generate_example(intent))

# Save as JSONL (UTF-8)
output_path = Path("data/dataset.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for example in examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"✅ Generated {len(examples)} examples at {output_path}")
