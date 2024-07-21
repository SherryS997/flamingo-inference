import requests
import json

def chat_with_audio(audio_file, dialogue):
    """
    Send a request to the chat inference API and return the response.
    
    :param audio_file: Path to the audio file
    :param dialogue: List of dialogue turns
    :return: The model's response
    """
    url = "http://localhost:5000/chat"
    
    payload = {
        "audio_file": audio_file,
        "dialogue": dialogue
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Example usage
    audio_file = "audioset/eval_segments/22khz/Y0bRUkLsttto.wav"
    dialogue = [
        {"user": "What genre does this music belong to?"},
        {"user": "Can you describe the vocals in this track?"}
    ]
    
    print("Sending request to chat inference API...")
    response = chat_with_audio(audio_file, dialogue)
    
    if response:
        print("\nAPI Response:")
        print(response)
    else:
        print("\nFailed to get a response from the API.")
    
    # Example of a multi-turn conversation
    print("\nStarting a multi-turn conversation...")
    audio_file = "audioset/eval_segments/22khz/YXyktNsq4SZU.wav"
    dialogue = [
        {"user": "Can you briefly explain what you hear in the audio?"}
    ]
    
    for _ in range(3):  # Simulate a 3-turn conversation
        response = chat_with_audio(audio_file, dialogue)
        if response:
            print(f"\nAPI Response: {response}")
            user_input = input("Your next question (or press Enter to end): ")
            if not user_input:
                break
            dialogue.append({"user": user_input})
        else:
            print("Failed to get a response from the API.")
            break

if __name__ == "__main__":
    main()