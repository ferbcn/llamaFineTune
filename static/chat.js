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

            currentResponse = document.createElement('div');
            currentResponse.className = 'message bot-message';
            messagesDiv.appendChild(currentResponse);

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                console.log(chunk);

                const chunkDiv = document.createElement("span");

                // Handle Markdown-style formatting
                // Order matters: handle triple before double before single
                let formattedChunk = chunk
                    // Bold-italic: ***text***
                    .replace(/\*\*\*([^\*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
                    // Bold: **text**
                    .replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>')
                    // Italic: *text*
                    .replace(/\*([^\*]+)\*/g, '<em>$1</em>');
                
                // Convert newlines to <br> tags
                formattedChunk = formattedChunk.replace(/\n/g, '<br>');
                
                // Use innerHTML instead of textContent to render HTML tags
                chunkDiv.innerHTML = formattedChunk;
                currentResponse.appendChild(chunkDiv);
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
            // Convert newlines to <br> tags for user messages too
            userDiv.innerHTML = message.replace(/\n/g, '<br>');
            messagesDiv.appendChild(userDiv);

            // Send message and handle response
            await sendMessage(message);

            // Clear input
            input.value = '';
        }
    });

})