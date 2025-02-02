document.addEventListener('DOMContentLoaded', function() {

    const messagesDiv = document.getElementById('messages');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');

    async function sendMessage(message) {
        // Get or create message container
        const chatContainer = document.getElementById('chat-container');

        // Create a new message element for the response
        const responseDiv = document.createElement('div');
        responseDiv.className = 'message assistant';
        chatContainer.appendChild(responseDiv);

        try {
            // Create EventSource for SSE connection
            const response = await fetch('/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message})
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));

                        switch (data.type) {
                            case 'start':
                                // Optional: Show processing message
                                currentResponse = document.createElement('div');
                                currentResponse.className = 'message bot-message';
                                messagesDiv.appendChild(currentResponse);
                                break;
                            case 'chunk':
                                // Append new token to response
                                currentResponse.innerHTML += data.content;
                                break;
                            case 'end':
                                // Optional: Handle completion
                                break;
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            responseDiv.textContent = 'Error: Failed to get response';
        }
    }

    // Example HTML form handler
    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const input = document.getElementById('message-input');
        const message = messageInput.value.trim();

        if (message) {
            // Create user message element
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.textContent = message;
            messagesDiv.appendChild(userDiv);

            // Send message and handle response
            await sendMessage(message);

            // Clear input
            input.value = '';
        }
    });

})