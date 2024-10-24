from openai import OpenAI
import json

# Replace 'your-api-key' with your actual OpenAI API key
client = OpenAI(api_key= 'insert your key')


def text_to_json(text, id):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts text descriptions into JSON format."},
                {"role": "user", "content": f"Convert the following text into JSON with properties. Assume that if feature is not mentioned it is absent or is of the lowest quality. Bathroom count starts from one. Properties: 'id' (integer), 'balcony' (boolean), 'bathroom_num' (integer), 'wardrobe' (boolean), 'view' (boolean), 'furnished' (boolean), 'appliances' (boolean), 'floor_heating' (boolean), 'air_conditioning' (boolean), 'parking' (boolean), 'security_features' (boolean), and 'renovation_quality' (integer from 1 to 5, where 1 means nothing and 5 means excellent): {text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        json_response = json.loads(response.choices[0].message.content)
        json_response['id'] = id  # Add the ID to the JSON response
        return json.dumps(json_response)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

