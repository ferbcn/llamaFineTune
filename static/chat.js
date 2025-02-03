document.addEventListener('DOMContentLoaded', function() {

    const messagesDiv = document.getElementById('messages');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const modelSelector = document.getElementById('model-selector');

    // Fetch available models when page loads
    async function loadModels() {
        try {
            const response = await fetch('/get-models');
            const models = await response.json();
            
            // Clear loading option
            modelSelector.innerHTML = '';
            
            // Add options for each model
            Object.entries(models).forEach(([key, value]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = key;
                modelSelector.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading models:', error);
            modelSelector.innerHTML = '<option value="">Error loading models</option>';
        }
    }

    // Load models when page loads
    loadModels();

    // Add this new function after loadModels()
    async function loadOptions() {
        try {
            const response = await fetch('/get-options');
            const options = await response.json();

            // Create inference options container
            const selectContainer = document.querySelector('.inference-options-container');

            // Create sliders for each option
            options.forEach(option => {
                const sliderContainer = document.createElement('div');
                sliderContainer.className = 'slider-container';
                
                // Create label and span
                const label = document.createElement('label');
                label.htmlFor = option.slider_id;

                const span = document.createElement('span');
                span.id = option.label_id;
                span.classList.add("slider-value");

                span.textContent = option.value;
                label.textContent = option.label + ": ";
                label.appendChild(span);
                
                // Create slider
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.id = option.slider_id;
                slider.min = option.min;
                slider.max = option.max;
                slider.step = option.step;
                slider.value = option.value; // Set the initial value before adding to DOM
                
                // Add event listener to slider
                slider.addEventListener('input', function() {
                    span.textContent = this.value;
                });
                
                // Append elements
                sliderContainer.appendChild(label);
                sliderContainer.appendChild(slider);
                selectContainer.appendChild(sliderContainer);
            });
            
        } catch (error) {
            console.error('Error loading options:', error);
        }
    }

    // Update the initialization section to call both functions
    loadModels();
    loadOptions();

    async function sendMessage(message) {
        // Get or create message container
        const chatContainer = document.getElementById('chat-container');

        // Create a new message element for the response
        const responseDiv = document.createElement('div');
        responseDiv.className = 'message assistant';
        chatContainer.appendChild(responseDiv);

        // get token count
        const tokenCount = parseInt(document.getElementById('token-slider').value);

        try {
            // Get all slider values
            const sliderValues = {};
            document.querySelectorAll('.slider-container input[type="range"]').forEach(slider => {
                const id = slider.id.replace('-slider', '');
                // Convert slider values to numbers and handle floating point values
                sliderValues[id] = Number(slider.value);
            });

            const response = await fetch('/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    model: modelSelector.value,
                    ...sliderValues
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            let currentResponse = null;  // Move declaration outside and initialize as null

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                console.log(chunk);

                // Create response div on first chunk
                if (!currentResponse) {
                    currentResponse = document.createElement('div');
                    currentResponse.className = 'message bot-message';
                    
                    // Add model indicator for bot response
                    const modelIndicator = document.createElement('div');
                    modelIndicator.className = 'model-indicator';
                    modelIndicator.textContent = `Response from: ${modelSelector.value}`;
                    currentResponse.appendChild(modelIndicator);
                    
                    messagesDiv.appendChild(currentResponse);
                }

                const chunkDiv = document.createElement("span");

                // Handle Markdown-style formatting
                // Order matters: handle more specific patterns first
                let formattedChunk = chunk
                    // Headers: ## Text
                    .replace(/##\s+([^\n]+)/g, '<h2>$1</h2>')
                    // LaTeX-style boxed content: \boxed{text}
                    .replace(/\\\boxed{([^}]+)}/g, '<div class="boxed">$1</div>')
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
            
            // Create model indicator
            const modelIndicator = document.createElement('div');
            modelIndicator.className = 'model-indicator';
            modelIndicator.textContent = `Using model: ${modelSelector.value}`;
            
            // Add message content with newlines converted to <br>
            const messageContent = document.createElement('div');
            messageContent.innerHTML = message.replace(/\n/g, '<br>');
            
            // Append both to the user message div
            userDiv.appendChild(modelIndicator);
            userDiv.appendChild(messageContent);
            messagesDiv.appendChild(userDiv);

            // Send message and handle response
            await sendMessage(message);

            // Clear input
            input.value = '';
        }
    });

})