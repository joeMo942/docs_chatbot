document.addEventListener("DOMContentLoaded", () => {
    const chatHistory = document.getElementById("chat-history");
    const messageInput = document.getElementById("message-input");
    const inputForm = document.getElementById("input-form");
    const sendButton = document.getElementById("send-button");
    const typingIndicator = document.getElementById("typing-indicator");

    // --- Auto-resize input textarea ---
    // This makes the input box grow as you type
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto'; // Reset height
        // 24px is vertical padding, 150px is a reasonable max height
        const newHeight = Math.min(messageInput.scrollHeight, 150);
        messageInput.style.height = (newHeight) + 'px';
    });

    // --- Handle form submission ---
    inputForm.addEventListener("submit", async (e) => {
        e.preventDefault(); // Prevent default form submission
        const question = messageInput.value.trim();

        if (question === "") {
            return; // Don't send empty messages
        }

        // Disable input and show typing indicator
        setChatLoading(true);

        // Add user message to chat history
        addMessageToHistory(question, "user");
        messageInput.value = ""; // Clear input
        messageInput.style.height = '44px'; // Reset height after send

        try {
            // Send request to the backend
            const response = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Handle the streaming response
            await handleStreamingResponse(response);

        } catch (error) {
            console.error("Error asking question:", error);
            // Show a formatted error message in the chat
            const botMsgElement = addMessageToHistory("", "bot", "error-message");
            botMsgElement.innerHTML = `<p>Error: Could not get a response from the server.</p>`;
        } finally {
            // Re-enable input and hide typing indicator
            setChatLoading(false);
        }
    });

    /**
     * Toggles the loading state of the chat input.
     * @param {boolean} isLoading - Whether the chat is loading.
     */
    function setChatLoading(isLoading) {
        if (isLoading) {
            typingIndicator.style.display = "flex";
            messageInput.disabled = true;
            sendButton.disabled = true;
            scrollToBottom();
        } else {
            typingIndicator.style.display = "none";
            messageInput.disabled = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    /**
     * Adds a new message bubble to the chat history.
     * @param {string} text - The message text.
     * @param {'user' | 'bot'} sender - The sender of the message.
     * @param {string} [extraClass] - Optional extra CSS class for the message element.
     * @returns {HTMLElement} The created message element.
     */
    function addMessageToHistory(text, sender, extraClass = '') {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        if (extraClass) {
            messageElement.classList.add(extraClass);
        }

        // We set the initial content. For bots, this will be formatted later.
        if (sender === 'user') {
            messageElement.textContent = text;
        }

        // Insert the new message *before* the typing indicator
        chatHistory.insertBefore(messageElement, typingIndicator);
        scrollToBottom();
        return messageElement;
    }

    /**
     * Handles the streaming response from the server.
     * @param {Response} response - The fetch response object.
     */
    async function handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";

        // Create a new bot message element to stream into
        const botMsgElement = addMessageToHistory("", "bot");

        // We add a <p> tag to stream into temporarily.
        // It will be replaced by the fully formatted content after the stream ends.
        let streamingP = document.createElement("p");
        botMsgElement.appendChild(streamingP);

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            const chunk = decoder.decode(value, { stream: true });
            fullResponse += chunk;
            streamingP.textContent = fullResponse; // Update the <p> tag
            scrollToBottom();
        }

        // Once streaming is done, format the full response
        formatBotResponse(botMsgElement, fullResponse);
    }

    /**
     * Formats the bot's final response, separating text and code blocks.
     * @param {HTMLElement} botMsgElement - The message bubble element to fill.
     * @param {string} text - The full response text from the bot.
     */
    function formatBotResponse(botMsgElement, text) {
        // Clear the temporary streaming content
        botMsgElement.innerHTML = "";

        // Split the text by markdown code blocks (```)
        const parts = text.split("```");

        parts.forEach((part, index) => {
            if (part.trim() === "") return;

            if (index % 2 === 0) {
                // This is plain text
                const p = document.createElement("p");
                p.textContent = part.trim();
                botMsgElement.appendChild(p);
            } else {
                // This is code. The first line might be the language (e.g., "cpp")
                const pre = document.createElement("pre");
                const code = document.createElement("code");

                let codeContent = part;
                const firstNewline = part.indexOf('\n');

                // Check if there is a language hint (e.g., "cpp")
                if (firstNewline !== -1) {
                    const language = part.substring(0, firstNewline).trim();
                    // Check if the language is a single word (like 'cpp', 'python', 'java')
                    if (language.length > 0 && !language.includes(" ")) {
                        code.classList.add(`language-${language}`);
                        codeContent = part.substring(firstNewline + 1); // Get text *after* the language hint
                    }
                }

                code.textContent = codeContent.trim();
                pre.appendChild(code);
                botMsgElement.appendChild(pre);
            }
        });

        // Handle case where bot just says "I do not have that information..."
        if (parts.length === 1 && text.includes("I do not have that information")) {
            botMsgElement.classList.add("error-message");
        }

        scrollToBottom();
    }

    /**
     * Scrolls the chat history to the bottom.
     */
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Add an initial welcome message
    const welcomeMsg = addMessageToHistory("", "bot");
    formatBotResponse(welcomeMsg, "Hello! I am DocuBot. Ask me anything about your C++ or Python documentation.");
    setChatLoading(false); // Make sure input is enabled at start
});