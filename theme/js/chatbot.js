/**
 * Prof. Keras - AI Documentation Assistant
 * Floating chatbot powered by Gemini for keras.io
 */
(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────────
  var isOpen = false;
  var docsIndex = null;
  var docsIndexLoading = false;
  var isGenerating = false;
  var conversationHistory = [];

  // ── Config ─────────────────────────────────────────────────────────
  var GEMINI_MODEL = "gemini-2.5-pro";
  var GEMINI_API_URL =
    "https://generativelanguage.googleapis.com/v1beta/models/" +
    GEMINI_MODEL +
    ":generateContent";
  var MAX_CONTEXT_CHUNKS = 12;
  var MAX_HISTORY_TURNS = 16;

  var SYSTEM_PROMPT =
    "You are Prof. Keras, a friendly and knowledgeable AI assistant specialized in the Keras deep learning library and its ecosystem (Keras Hub, Keras Tuner, Keras RS).\n\n" +
    "Your role:\n" +
    "- Help users understand Keras concepts, APIs, and best practices\n" +
    "- Assist with debugging Keras code\n" +
    "- Guide users to the right APIs and patterns for their tasks\n" +
    "- Provide concise, practical code examples when helpful\n\n" +
    "Guidelines:\n" +
    "- Answer based on the documentation context provided. If the context doesn't cover the question, say so honestly.\n" +
    "- Keep answers concise but complete. Prefer code examples over lengthy explanations.\n" +
    "- Use markdown formatting: backticks for code, **bold** for emphasis, bullet lists for steps.\n" +
    "- When referencing a documentation page, mention its title so the user can search for it.\n" +
    "- For Keras 3, remember it supports JAX, TensorFlow, and PyTorch backends.\n" +
    "- Be friendly and encouraging, especially with beginners.";

  // ── API Key ────────────────────────────────────────────────────────
  function getApiKey() {
    return (window.KERAS_CHATBOT_CONFIG && window.KERAS_CHATBOT_CONFIG.apiKey) || "";
  }

  // ── UI Creation ────────────────────────────────────────────────────
  function createWidget() {
    // Floating button
    var btn = document.createElement("button");
    btn.id = "keras-chatbot-btn";
    btn.setAttribute("aria-label", "Ask Prof. Keras");
    btn.title = "Ask Prof. Keras";
    btn.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' +
      '<path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>' +
      '<path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>' +
      "</svg>";
    btn.addEventListener("click", toggleChat);

    // Chat window
    var win = document.createElement("div");
    win.id = "keras-chatbot-window";
    win.innerHTML =
      '<div class="chatbot-header">' +
      '  <div class="chatbot-header-title">' +
      '    <img src="' + getBaseUrl() + 'img/prof_keras.png" alt="Prof. Keras" />' +
      "    <div>" +
      "      <div>Prof. Keras</div>" +
      '      <div class="chatbot-header-subtitle">Keras Documentation Assistant</div>' +
      "    </div>" +
      "  </div>" +
      '  <button class="chatbot-close-btn" aria-label="Close chat">&times;</button>' +
      "</div>" +
      '<div class="chatbot-messages" id="chatbot-messages">' +
      '  <div class="chatbot-msg chatbot-msg-bot">' +
      '    <div class="chatbot-msg-bubble">' +
      "      <p>Hi! I'm <strong>Prof. Keras</strong>. Ask me anything about Keras — APIs, guides, code examples, or debugging help.</p>" +
      "    </div>" +
      "  </div>" +
      "</div>" +
      '<div class="chatbot-input-area">' +
      '  <input class="chatbot-input" id="chatbot-input" type="text" placeholder="Ask a question about Keras..." autocomplete="off" />' +
      '  <button class="chatbot-send-btn" id="chatbot-send" aria-label="Send message">' +
      '    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>' +
      "  </button>" +
      "</div>" +
      '<div class="chatbot-powered">Powered by Gemini</div>';

    document.body.appendChild(btn);
    document.body.appendChild(win);

    // Event listeners
    win.querySelector(".chatbot-close-btn").addEventListener("click", toggleChat);
    win.querySelector("#chatbot-send").addEventListener("click", handleSend);
    win.querySelector("#chatbot-input").addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });
  }

  function getBaseUrl() {
    // Try to infer the base URL from existing elements
    var logoLink = document.querySelector("a[href] img.nav__logo");
    if (logoLink && logoLink.parentElement) {
      var href = logoLink.parentElement.getAttribute("href");
      if (href) return href;
    }
    return "/";
  }

  // ── Toggle Chat ────────────────────────────────────────────────────
  function toggleChat() {
    isOpen = !isOpen;
    var win = document.getElementById("keras-chatbot-window");
    var btn = document.getElementById("keras-chatbot-btn");

    if (isOpen) {
      win.classList.add("chatbot-visible");
      btn.classList.add("chatbot-open");
      document.getElementById("chatbot-input").focus();
      if (!docsIndex && !docsIndexLoading) {
        loadDocsIndex();
      }
    } else {
      win.classList.remove("chatbot-visible");
      btn.classList.remove("chatbot-open");
    }
  }

  // ── Load Documentation Index ───────────────────────────────────────
  function loadDocsIndex() {
    docsIndexLoading = true;
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/docs_index.json", true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4) {
        docsIndexLoading = false;
        if (xhr.status === 200) {
          try {
            docsIndex = JSON.parse(xhr.responseText);
          } catch (e) {
            console.error("Prof. Keras: Failed to parse docs index", e);
          }
        } else {
          console.error("Prof. Keras: Failed to load docs index, status:", xhr.status);
        }
      }
    };
    xhr.send();
  }

  // ── Search Documentation ───────────────────────────────────────────
  var STOP_WORDS = {
    the: 1, a: 1, an: 1, and: 1, or: 1, but: 1, in: 1, on: 1, at: 1, to: 1,
    for: 1, of: 1, is: 1, it: 1, be: 1, as: 1, by: 1, this: 1, that: 1,
    from: 1, with: 1, are: 1, was: 1, were: 1, been: 1, has: 1, have: 1,
    had: 1, do: 1, does: 1, did: 1, will: 1, would: 1, can: 1, could: 1,
    should: 1, may: 1, might: 1, not: 1, no: 1, so: 1, if: 1, my: 1,
    me: 1, we: 1, you: 1, your: 1, they: 1, them: 1, what: 1, which: 1,
    who: 1, how: 1, when: 1, where: 1, why: 1, all: 1, each: 1, some: 1,
    any: 1, i: 1, am: 1, get: 1, use: 1, using: 1, want: 1, need: 1,
  };

  function searchDocs(query) {
    if (!docsIndex || docsIndex.length === 0) return [];

    var queryLower = query.toLowerCase();
    var words = queryLower.split(/\s+/);
    var keywords = [];
    for (var i = 0; i < words.length; i++) {
      var w = words[i].replace(/[^a-z0-9_\.]/g, "");
      if (w.length > 1 && !STOP_WORDS[w]) {
        keywords.push(w);
      }
    }

    if (keywords.length === 0) return [];

    var scored = [];
    for (var j = 0; j < docsIndex.length; j++) {
      var chunk = docsIndex[j];
      var titleLower = chunk.title.toLowerCase();
      var contentLower = chunk.content.toLowerCase();
      var score = 0;

      for (var k = 0; k < keywords.length; k++) {
        var kw = keywords[k];
        // Title matches are weighted higher
        var titleIdx = titleLower.indexOf(kw);
        if (titleIdx !== -1) score += 10;

        // Content matches
        var contentIdx = 0;
        var searchFrom = 0;
        while ((contentIdx = contentLower.indexOf(kw, searchFrom)) !== -1) {
          score += 1;
          searchFrom = contentIdx + kw.length;
        }
      }

      // Bonus for exact phrase match
      if (contentLower.indexOf(queryLower) !== -1) {
        score += 20;
      }

      if (score > 0) {
        scored.push({ chunk: chunk, score: score });
      }
    }

    scored.sort(function (a, b) {
      return b.score - a.score;
    });

    // Deduplicate by URL (keep highest scored chunk per page, but allow multiple)
    var urlCounts = {};
    var results = [];
    for (var m = 0; m < scored.length && results.length < MAX_CONTEXT_CHUNKS; m++) {
      var url = scored[m].chunk.url;
      urlCounts[url] = (urlCounts[url] || 0) + 1;
      if (urlCounts[url] <= 3) {
        results.push(scored[m].chunk);
      }
    }

    return results;
  }

  // ── Send Message ───────────────────────────────────────────────────
  function handleSend() {
    var input = document.getElementById("chatbot-input");
    var message = input.value.trim();
    if (!message || isGenerating) return;

    var apiKey = getApiKey();
    if (!apiKey) {
      appendMessage(
        "bot",
        "I'm not configured yet. The Gemini API key is missing. Please check the build configuration."
      );
      return;
    }

    input.value = "";
    appendMessage("user", message);

    isGenerating = true;
    setSendEnabled(false);
    showTypingIndicator();

    // Search for relevant documentation
    var relevantDocs = searchDocs(message);
    var context = "";
    if (relevantDocs.length > 0) {
      var parts = [];
      for (var i = 0; i < relevantDocs.length; i++) {
        var doc = relevantDocs[i];
        parts.push("## " + doc.title + " (" + doc.url + ")\n" + doc.content);
      }
      context = parts.join("\n\n---\n\n");
    }

    // Add user message to history
    conversationHistory.push({
      role: "user",
      parts: [{ text: message }],
    });

    // Trim history
    if (conversationHistory.length > MAX_HISTORY_TURNS) {
      conversationHistory = conversationHistory.slice(-MAX_HISTORY_TURNS);
    }

    // Call Gemini
    callGemini(context, conversationHistory, function (err, response) {
      hideTypingIndicator();
      isGenerating = false;
      setSendEnabled(true);

      if (err) {
        appendMessage("bot", "Sorry, I encountered an error: " + err + ". Please try again.");
        return;
      }

      conversationHistory.push({
        role: "model",
        parts: [{ text: response }],
      });

      appendMessage("bot", response);
    });
  }

  // ── Gemini API Call ────────────────────────────────────────────────
  function callGemini(context, history, callback) {
    var apiKey = getApiKey();
    var url = GEMINI_API_URL + "?key=" + apiKey;

    var systemText = SYSTEM_PROMPT;
    if (context) {
      systemText +=
        "\n\n--- RELEVANT DOCUMENTATION ---\n\n" +
        context +
        "\n\n--- END DOCUMENTATION ---";
    }

    var body = {
      system_instruction: {
        parts: [{ text: systemText }],
      },
      contents: history,
      generationConfig: {
        temperature: 0.4,
        maxOutputTokens: 2048,
      },
    };

    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) return;

      if (xhr.status === 200) {
        try {
          var data = JSON.parse(xhr.responseText);
          if (
            data.candidates &&
            data.candidates[0] &&
            data.candidates[0].content &&
            data.candidates[0].content.parts &&
            data.candidates[0].content.parts[0]
          ) {
            callback(null, data.candidates[0].content.parts[0].text);
          } else {
            callback("Unexpected response format");
          }
        } catch (e) {
          callback("Failed to parse response");
        }
      } else {
        var errMsg = "API request failed (status " + xhr.status + ")";
        try {
          var errData = JSON.parse(xhr.responseText);
          if (errData.error && errData.error.message) {
            errMsg = errData.error.message;
          }
        } catch (e) {
          // ignore parse error
        }
        callback(errMsg);
      }
    };
    xhr.onerror = function () {
      callback("Network error. Please check your connection.");
    };
    xhr.send(JSON.stringify(body));
  }

  // ── UI Helpers ─────────────────────────────────────────────────────
  function appendMessage(sender, text) {
    var messages = document.getElementById("chatbot-messages");
    var msgDiv = document.createElement("div");
    msgDiv.className = "chatbot-msg chatbot-msg-" + sender;

    var bubble = document.createElement("div");
    bubble.className = "chatbot-msg-bubble";

    if (sender === "bot") {
      bubble.innerHTML = renderMarkdown(text);
    } else {
      bubble.textContent = text;
    }

    msgDiv.appendChild(bubble);
    messages.appendChild(msgDiv);
    messages.scrollTop = messages.scrollHeight;
  }

  function showTypingIndicator() {
    var messages = document.getElementById("chatbot-messages");
    var existing = document.getElementById("chatbot-typing");
    if (existing) return;

    var div = document.createElement("div");
    div.id = "chatbot-typing";
    div.className = "chatbot-msg chatbot-msg-bot";
    div.innerHTML =
      '<div class="chatbot-msg-bubble">' +
      '<div class="chatbot-typing-indicator">' +
      "<span></span><span></span><span></span>" +
      "</div></div>";
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  function hideTypingIndicator() {
    var el = document.getElementById("chatbot-typing");
    if (el) el.remove();
  }

  function setSendEnabled(enabled) {
    var btn = document.getElementById("chatbot-send");
    if (btn) btn.disabled = !enabled;
  }

  // ── Markdown Renderer ──────────────────────────────────────────────
  function renderMarkdown(text) {
    if (!text) return "";

    // Escape HTML first
    text = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // Code blocks (fenced)
    text = text.replace(
      /```(\w*)\n([\s\S]*?)```/g,
      function (match, lang, code) {
        // Unescape HTML inside code blocks to preserve formatting
        code = code.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
        // Then re-escape for display
        code = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return '<pre><code class="language-' + (lang || "python") + '">' + code + "</code></pre>";
      }
    );

    // Inline code
    text = text.replace(/`([^`\n]+)`/g, "<code>$1</code>");

    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic
    text = text.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");

    // Links
    text = text.replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );

    // Unordered lists
    text = text.replace(/^[\-\*] (.+)/gm, "<li>$1</li>");
    text = text.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered lists
    text = text.replace(/^\d+\. (.+)/gm, "<li>$1</li>");

    // Paragraphs: split on double newlines
    var blocks = text.split(/\n\n+/);
    var result = [];
    for (var i = 0; i < blocks.length; i++) {
      var block = blocks[i].trim();
      if (!block) continue;
      if (
        block.indexOf("<pre>") === 0 ||
        block.indexOf("<ul>") === 0 ||
        block.indexOf("<ol>") === 0 ||
        block.indexOf("<li>") === 0
      ) {
        result.push(block);
      } else {
        // Convert single newlines to <br> within paragraphs
        block = block.replace(/\n/g, "<br>");
        result.push("<p>" + block + "</p>");
      }
    }

    return result.join("");
  }

  // ── Initialize ─────────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", createWidget);
  } else {
    createWidget();
  }
})();
