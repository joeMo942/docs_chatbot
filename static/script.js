document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = userInput.value.trim();
        if (!question) return;

        // Display user's question
        addMessageToChat(question, 'user');
        
        // Clear input and show thinking indicator
        userInput.value = '';
        const botMessageElement = addMessageToChat('...', 'bot');

        try {
            // Send question to FastAPI backend
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                botMessageElement.innerHTML = `<p class="error-message">Error: ${response.statusText}</p>`;
                return;
            }

            // --- Handle the streaming response ---
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let botResponse = '';

            // Clear the '...'
            botMessageElement.innerHTML = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                botResponse += chunk;
                botMessageElement.innerHTML = parseMarkdown(botResponse); // Render markdown
                
                // Auto-scroll to bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

        } catch (error) {
            botMessageElement.innerHTML = `<p class="error-message">Error: Could not connect to the server. ${error}</p>`;
        }
    });

    // Helper function to add a new message to the chat UI
    function addMessageToChat(message, sender) {
        const messageWrapper = document.createElement('div');
        const messageBubble = document.createElement('div');
        
        messageWrapper.classList.add('message-wrapper');
        messageBubble.classList.add('message-bubble');
        
        if (sender === 'user') {
            messageWrapper.classList.add('message-wrapper-user');
            messageBubble.classList.add('message-bubble-user');
            messageBubble.textContent = message;
        } else {
            messageWrapper.classList.add('message-wrapper-bot');
            messageBubble.classList.add('message-bubble-bot');
            messageBubble.innerHTML = message; // Allow HTML for streaming/errors
        }
        
        messageWrapper.appendChild(messageBubble);
        chatContainer.appendChild(messageWrapper);
        
        // Auto-scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        return messageBubble; // Return the bubble element for streaming
    }
    
    // Simple markdown parser (for code blocks)
    function parseMarkdown(text) {
        // Convert ```code``` to <pre><code>...</code></pre>
        text = text.replace(/```([\s\S]*?)```/g, (match, code) => {
            // Escape HTML in code
            const safeCode = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<pre><code>${safeCode.trim()}</code></pre>`;
        });
        // Convert `code` to <code>...</code>
        text = text.replace(/`([\s\S]*?)`/g, (match, code) => {
            const safeCode = code.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<code>${safeCode}</code>`;
        });
        return text.replace(/\n/g, '<br>'); // Handle newlines
    }
});