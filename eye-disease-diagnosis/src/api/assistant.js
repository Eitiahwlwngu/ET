
const API_BASE_URL = 'http://localhost:3000';

export const sendMessageToAssistant = async (message) => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
      }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('Error sending message to assistant:', error);
    return '抱歉，我无法回答您的问题，请稍后再试。';
  }
};