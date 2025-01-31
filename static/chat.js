document.addEventListener('DOMContentLoaded', function() {
    const messagesDiv = document.getElementById('messages');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    let ws;
    let currentResponse = null;

    function connect() {
        ws = new WebSocket(`ws://${window.location.host}/ws-chat`);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.type === 'start') {
                // Create new response container
                currentResponse = document.createElement('div');
                currentResponse.className = 'message bot-message';
                messagesDiv.appendChild(currentResponse);
            } else if (data.type === 'chunk') {
                // Append to current response
                if (currentResponse) {
                    currentResponse.innerHTML += data.content;
                }
            } else if (data.type === 'end') {
                currentResponse = null;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        };

        ws.onclose = function() {
            // Reconnect on close
            setTimeout(connect, 1000);
        };
    }

    messageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = messageInput.value;
        if (message && ws.readyState === WebSocket.OPEN) {
            // Add user message to chat
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.textContent = message;
            messagesDiv.appendChild(userDiv);

            // Send to server
            ws.send(JSON.stringify({ message: message }));

            // Clear input
            messageInput.value = '';
        }
    });

    // Initial connection
    connect();
});