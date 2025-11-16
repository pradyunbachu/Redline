import React, { useState, useRef, useEffect } from "react";
import "./ChatAgent.css";

const ChatAgent = ({ damageResults, originalImage }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true); // Toggle for speech
  const [zoomedImages, setZoomedImages] = useState([]); // Array of {image, boxIndex, info}
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioStreamRef = useRef(null); // Store the audio stream separately
  const audioChunksRef = useRef([]);
  const pendingRequestRef = useRef(false); // Prevent duplicate API calls

  const handleSendMessage = async (messageText, isVoiceInput = false) => {
    if (!messageText.trim() || isLoading || pendingRequestRef.current) return;

    const userMessage = messageText.trim();
    setInput("");
    setIsLoading(true);
    pendingRequestRef.current = true; // Mark request as pending

    // Add user message first
    const newMessages = [...messages, { role: "user", content: userMessage }];
    setMessages(newMessages);

    try {
      const response = await fetch("http://localhost:5001/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_history: messages.slice(-10), // Last 10 messages BEFORE current one
          damage_results: damageResults,
        }),
      });

      const data = await response.json();

      const assistantMessage = { role: "assistant", content: data.response };
      setMessages((prev) => [...prev, assistantMessage]);

      // If response includes box indices, fetch and display the zoomed images
      if (
        data.box_indices !== undefined &&
        data.box_indices !== null &&
        originalImage &&
        damageResults
      ) {
        const boxIndices = Array.isArray(data.box_indices)
          ? data.box_indices
          : [data.box_indices];
        const instances = damageResults.damage_instances || [];

        // Fetch cropped images for all detected boxes
        const imagePromises = boxIndices
          .filter((boxIndex) => boxIndex >= 0 && boxIndex < instances.length)
          .map(async (boxIndex) => {
            const instance = instances[boxIndex];
            const bbox = instance.bbox;

            if (bbox && bbox.length >= 4) {
              try {
                const cropResponse = await fetch(
                  "http://localhost:5001/api/crop-damage-box",
                  {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                      image: originalImage,
                      bbox: bbox,
                    }),
                  }
                );

                const cropData = await cropResponse.json();
                if (cropData.success) {
                  return {
                    image: cropData.cropped_image,
                    boxIndex: boxIndex,
                    info: {
                      part_name: instance.part_name,
                      damage_class: instance.damage_class,
                      severity_class: instance.severity_class,
                      cost: instance.cost_estimate?.final_cost || 0,
                    },
                  };
                }
              } catch (error) {
                console.error(
                  `Error cropping image for box ${boxIndex}:`,
                  error
                );
              }
            }
            return null;
          });

        // Wait for all images to load
        const zoomedImagesData = await Promise.all(imagePromises);
        const validImages = zoomedImagesData.filter((img) => img !== null);

        if (validImages.length > 0) {
          setZoomedImages(validImages);
        } else {
          setZoomedImages([]);
        }
      } else {
        // Clear zoomed images if not asking about specific boxes
        setZoomedImages([]);
      }

      // Only speak the response if it came from voice input
      console.log(
        "📨 Response received, isVoiceInput:",
        isVoiceInput,
        "speechEnabled:",
        speechEnabled
      );
      if (isVoiceInput && speechEnabled) {
        console.log(
          "🔊 Voice input detected - speaking response:",
          data.response.substring(0, 50)
        );
        // Small delay to ensure message is rendered and user interaction is registered
        setTimeout(() => {
          speakText(data.response);
        }, 100);
      } else {
        console.log("🔇 Text input or speech disabled - not speaking");
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
        },
      ]);
    } finally {
      setIsLoading(false);
      pendingRequestRef.current = false; // Clear pending flag
    }
  };

  // Simple, clean text-to-speech function
  const speakText = (text) => {
    console.log("🔊 SPEAK TEXT CALLED:", text.substring(0, 50));

    if (!text || !text.trim()) {
      console.log("❌ No text to speak");
      return;
    }

    if (!("speechSynthesis" in window)) {
      console.error("❌ Speech synthesis not supported");
      alert("Text-to-speech is not supported in your browser.");
      return;
    }

    console.log("✓ Speech synthesis available");

    // Stop any current speech ONLY if something is actually speaking
    const wasSpeaking =
      window.speechSynthesis.speaking || window.speechSynthesis.pending;
    if (wasSpeaking) {
      console.log("⏹ Stopping current speech...");
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      // Wait a bit for cancel to complete before starting new speech
      setTimeout(() => {
        doActualSpeak(text);
      }, 200);
    } else {
      // Nothing is speaking, so we can speak immediately
      doActualSpeak(text);
    }
  };

  const doActualSpeak = (text) => {
    console.log("🎙️ doActualSpeak called");

    // Clean the text and add natural pauses
    let cleanText = text
      .replace(/\*\*/g, "")
      .replace(/\*/g, "")
      .replace(/#{1,6}\s/g, "")
      .replace(/```[\s\S]*?```/g, "")
      .replace(/`[^`]+`/g, "")
      .replace(/\n{2,}/g, ". ") // Multiple newlines become period + space
      .replace(/\n/g, " ") // Single newlines become spaces
      .replace(/\.\s+\./g, ".") // Remove double periods
      .replace(/\s+/g, " ") // Multiple spaces become single space
      .trim();

    // Add natural pauses after commas and periods for more natural speech
    cleanText = cleanText
      .replace(/,\s*/g, ", ") // Ensure space after commas
      .replace(/\.\s*/g, ". ") // Ensure space after periods
      .replace(/\?\s*/g, "? ") // Ensure space after question marks
      .replace(/!\s*/g, "! "); // Ensure space after exclamation marks

    if (!cleanText) {
      console.log("❌ No clean text after processing");
      return;
    }

    console.log("📝 Clean text:", cleanText.substring(0, 100));

    // Function to actually speak (called after voices are loaded)
    const doSpeak = () => {
      // Create utterance FIRST before checking if something is speaking
      const utterance = new SpeechSynthesisUtterance(cleanText);
      // More natural speech settings
      utterance.rate = 1.2; // Faster speech rate
      utterance.pitch = 0.95; // Slightly lower pitch sounds more natural
      utterance.volume = 1.0;

      // Get a voice
      const voices = window.speechSynthesis.getVoices();
      console.log("🎤 Available voices:", voices.length);

      if (voices.length > 0) {
        // Prefer natural-sounding voices - prioritize high-quality, conversational voices
        const preferredVoice =
          voices.find(
            (v) =>
              // macOS/iOS natural voices
              v.name.includes("Samantha") ||
              v.name.includes("Alex") ||
              v.name.includes("Victoria") ||
              v.name.includes("Daniel") ||
              v.name.includes("Karen") ||
              v.name.includes("Moira") ||
              // Windows natural voices
              v.name.includes("Zira") ||
              v.name.includes("David") ||
              // Enhanced/Premium voices
              v.name.includes("Enhanced") ||
              v.name.includes("Premium") ||
              v.name.includes("Natural") ||
              // Avoid robotic/compact voices
              (v.lang.startsWith("en") &&
                !v.name.includes("Albert") &&
                !v.name.includes("Compact") &&
                !v.name.includes("Novelty") &&
                !v.name.includes("Cellos") &&
                !v.name.includes("Bells") &&
                !v.name.includes("Organ"))
          ) ||
          voices.find((v) => v.lang.startsWith("en")) ||
          voices[0];

        utterance.voice = preferredVoice;
        console.log("✓ Using voice:", preferredVoice.name, preferredVoice.lang);
      } else {
        console.warn("⚠ No voices available, using default");
      }

      // Set up event handlers BEFORE calling speak()
      utterance.onstart = () => {
        console.log("✅✅✅ SPEECH STARTED! Audio should be playing now.");
        setIsSpeaking(true);
      };

      utterance.onend = () => {
        console.log("✅ Speech ended - audio finished");
        setIsSpeaking(false);
      };

      utterance.onpause = () => {
        console.log("⏸ Speech paused");
      };

      utterance.onresume = () => {
        console.log("▶ Speech resumed");
      };

      utterance.onerror = (error) => {
        console.error("❌❌❌ SPEECH ERROR:", error.error, error);
        console.error("Error details:", {
          error: error.error,
          type: error.type,
          charIndex: error.charIndex,
          charLength: error.charLength,
        });
        setIsSpeaking(false);
        // Only alert for real errors, not "canceled"
        if (error.error !== "canceled") {
          alert(`Speech error: ${error.error}. Check console for details.`);
        } else {
          console.log("⚠ Speech was canceled (this is usually fine)");
        }
      };

      // Speak it - but first cancel any existing speech if needed
      console.log("🚀 Calling speechSynthesis.speak() NOW...");
      console.log(
        "📊 Before speak - speaking:",
        window.speechSynthesis.speaking,
        "pending:",
        window.speechSynthesis.pending
      );

      // Only cancel if something is actually speaking
      if (window.speechSynthesis.speaking) {
        console.log("⏹ Something is speaking, canceling it first...");
        window.speechSynthesis.cancel();
        // Wait a moment for cancel to complete
        setTimeout(() => {
          console.log(
            "📊 After cancel - speaking:",
            window.speechSynthesis.speaking,
            "pending:",
            window.speechSynthesis.pending
          );
          try {
            window.speechSynthesis.speak(utterance);
            console.log("✓ speak() called successfully after cancel");
          } catch (error) {
            console.error("❌ Exception calling speak():", error);
            alert("Error calling speech: " + error.message);
          }
        }, 100);
      } else {
        // Nothing is speaking, so we can speak immediately
        try {
          console.log("🎯 About to call speak() - utterance ready");

          // Try to resume speech synthesis in case it's paused
          if (window.speechSynthesis.paused) {
            console.log("▶ Speech synthesis was paused, resuming...");
            window.speechSynthesis.resume();
          }

          window.speechSynthesis.speak(utterance);
          console.log("✓ speak() called successfully (nothing was speaking)");

          // Force it to start if it's just queued
          setTimeout(() => {
            const isSpeaking =
              window.speechSynthesis.speaking || window.speechSynthesis.pending;
            console.log("📊 Immediate check (50ms):", {
              speaking: window.speechSynthesis.speaking,
              pending: window.speechSynthesis.pending,
              isSpeaking: isSpeaking,
            });

            // If it's pending but not speaking, try to force start
            if (
              window.speechSynthesis.pending &&
              !window.speechSynthesis.speaking
            ) {
              console.log(
                "⚠ Speech is pending but not started - trying to resume..."
              );
              window.speechSynthesis.resume();
            }
          }, 50);
        } catch (error) {
          console.error("❌ Exception calling speak():", error);
          alert("Error calling speech: " + error.message);
        }
      }

      // Check after a moment if it started
      setTimeout(() => {
        const isSpeaking =
          window.speechSynthesis.speaking || window.speechSynthesis.pending;
        console.log("📊 Speech status check after 500ms:", {
          speaking: window.speechSynthesis.speaking,
          pending: window.speechSynthesis.pending,
          isSpeaking: isSpeaking,
        });

        if (isSpeaking) {
          console.log("✓ Browser says it's speaking");
          if (
            !window.speechSynthesis.speaking &&
            window.speechSynthesis.pending
          ) {
            console.warn(
              "⚠ Speech is pending but not actually speaking - trying to resume..."
            );
            window.speechSynthesis.resume();
          }
          console.log("🔍 If you can't hear audio, check:");
          console.log("   1. System volume (not muted)");
          console.log(
            "   2. Browser tab audio (click the speaker icon in the tab)"
          );
          console.log("   3. Browser audio settings");
          console.log("   4. Try a different browser (Chrome/Edge work best)");
        } else {
          console.warn("⚠ Speech did not start - onstart never fired");
          console.log("🔍 Trying to diagnose...");
          console.log(
            "   - Speech synthesis available:",
            "speechSynthesis" in window
          );
          console.log(
            "   - Voices loaded:",
            window.speechSynthesis.getVoices().length
          );
          console.log(
            "   - Current speaking state:",
            window.speechSynthesis.speaking
          );
          alert(
            "Speech did not start. The browser may be blocking audio. Try:\n1. Check system volume\n2. Check browser audio settings\n3. Try refreshing the page\n4. Try a different browser (Chrome/Edge)"
          );
        }
      }, 500);

      // Also check after 1 second
      setTimeout(() => {
        if (window.speechSynthesis.speaking) {
          console.log(
            "📊 Still speaking after 1 second - audio should be playing"
          );
        }
      }, 1000);
    };

    // Ensure voices are loaded before speaking
    const voices = window.speechSynthesis.getVoices();
    if (voices.length === 0) {
      console.log("⏳ No voices yet, waiting for voices to load...");
      // Wait for voices to load
      const checkVoices = () => {
        const loadedVoices = window.speechSynthesis.getVoices();
        if (loadedVoices.length > 0) {
          console.log("✓ Voices loaded:", loadedVoices.length);
          doSpeak();
        } else {
          console.log("⏳ Still waiting for voices...");
          setTimeout(checkVoices, 100);
        }
      };

      // Set up voice change listener
      if (window.speechSynthesis.onvoiceschanged) {
        window.speechSynthesis.onvoiceschanged = () => {
          console.log("✓ Voices changed event fired");
          doSpeak();
        };
      }

      // Also check periodically
      setTimeout(checkVoices, 100);
      setTimeout(checkVoices, 500);
      setTimeout(checkVoices, 1000);
    } else {
      console.log("✓ Voices already loaded");
      doSpeak();
    }
  };

  const toggleSpeech = () => {
    const newState = !speechEnabled;
    if (speechEnabled && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setSpeechEnabled(newState);
  };

  useEffect(() => {
    // Check if browser supports MediaRecorder API for audio recording
    if (
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia &&
      typeof MediaRecorder !== "undefined"
    ) {
      setIsSupported(true);

      // Check current permission status (optional, for debugging)
      if (navigator.permissions && navigator.permissions.query) {
        navigator.permissions
          .query({ name: "microphone" })
          .then((result) => {
            console.log("Microphone permission status:", result.state);
          })
          .catch(() => {
            // Some browsers don't support this API, that's okay
          });
      }
    }

    // Load voices for speech synthesis
    if ("speechSynthesis" in window) {
      const loadVoices = () => {
        window.speechSynthesis.getVoices();
      };
      loadVoices();
      if (window.speechSynthesis.onvoiceschanged) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }
    }

    // Initialize with welcome message
    if (messages.length === 0 && damageResults) {
      setMessages([
        {
          role: "assistant",
          content:
            "Hi! I'm here to help answer any questions about your damage estimate. You can type your questions or click the microphone to speak. Feel free to ask about specific repairs, costs, parts, or anything else related to your vehicle assessment.",
        },
      ]);
      // Don't speak the welcome message automatically - wait for user interaction
    }

    // Cleanup
    return () => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        try {
          mediaRecorderRef.current.stop();
        } catch (e) {
          // Ignore errors on cleanup
        }
      }
      // Stop all audio tracks
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [damageResults]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    await handleSendMessage(input, false); // false = text input, don't speak
  };

  const handleVoiceClick = async () => {
    if (!isSupported) {
      alert(
        "Audio recording is not supported in your browser. Please use Chrome, Edge, or Safari."
      );
      return;
    }

    if (isListening) {
      // Stop recording and transcribe
      try {
        if (
          mediaRecorderRef.current &&
          mediaRecorderRef.current.state !== "inactive"
        ) {
          mediaRecorderRef.current.stop();
        }
      } catch (error) {
        console.error("Error stopping recording:", error);
        setIsListening(false);
      }
    } else {
      // Start recording
      try {
        // Check if we're on HTTPS or localhost (required for microphone access)
        const isSecureContext =
          window.isSecureContext ||
          window.location.protocol === "https:" ||
          window.location.hostname === "localhost" ||
          window.location.hostname === "127.0.0.1";

        if (!isSecureContext) {
          alert(
            "Microphone access requires HTTPS or localhost. Please access the site via HTTPS or localhost."
          );
          return;
        }

        // Check if MediaRecorder is supported
        if (typeof MediaRecorder === "undefined") {
          alert(
            "MediaRecorder is not supported in your browser. Please use a modern browser like Chrome, Edge, or Safari."
          );
          return;
        }

        // Request microphone permission with detailed error handling
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          });
        } catch (permError) {
          console.error("Permission error:", permError);
          let errorMessage = "Could not access microphone. ";

          if (
            permError.name === "NotAllowedError" ||
            permError.name === "PermissionDeniedError"
          ) {
            errorMessage +=
              "Please allow microphone access in your browser settings and try again.";
          } else if (
            permError.name === "NotFoundError" ||
            permError.name === "DevicesNotFoundError"
          ) {
            errorMessage +=
              "No microphone found. Please connect a microphone and try again.";
          } else if (
            permError.name === "NotReadableError" ||
            permError.name === "TrackStartError"
          ) {
            errorMessage +=
              "Microphone is being used by another application. Please close other apps using the microphone.";
          } else if (
            permError.name === "OverconstrainedError" ||
            permError.name === "ConstraintNotSatisfiedError"
          ) {
            errorMessage +=
              "Microphone settings are not supported. Please try a different browser.";
          } else {
            errorMessage += `Error: ${
              permError.message || permError.name
            }. Please check your browser settings.`;
          }

          alert(errorMessage);
          setIsListening(false);
          return;
        }

        // Determine the best audio format for the browser
        let mimeType = "audio/webm";
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/webm")) {
          mimeType = "audio/webm";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus";
        } else {
          // Fallback - let browser choose
          mimeType = "";
        }

        let mediaRecorder;
        try {
          if (mimeType) {
            mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
          } else {
            mediaRecorder = new MediaRecorder(stream);
          }
        } catch (recorderError) {
          console.error("Error creating MediaRecorder:", recorderError);
          stream.getTracks().forEach((track) => track.stop());
          audioStreamRef.current = null;
          alert(
            "Error initializing audio recorder. Please try a different browser."
          );
          setIsListening(false);
          return;
        }

        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          setIsListening(false);
          setInput("🎤 Transcribing...");

          // Get stream from ref for cleanup
          const currentStream = audioStreamRef.current;

          try {
            // Check if we have any audio data
            if (audioChunksRef.current.length === 0) {
              alert("No audio was recorded. Please try again.");
              setInput("");
              if (currentStream) {
                currentStream.getTracks().forEach((track) => track.stop());
                audioStreamRef.current = null;
              }
              return;
            }

            // Create audio blob
            const audioBlob = new Blob(audioChunksRef.current, {
              type: mimeType || "audio/webm",
            });

            // Check blob size (should be > 0)
            if (audioBlob.size === 0) {
              alert("Recorded audio is empty. Please try again.");
              setInput("");
              if (currentStream) {
                currentStream.getTracks().forEach((track) => track.stop());
                audioStreamRef.current = null;
              }
              return;
            }

            // Send to backend for transcription using Groq Whisper
            const formData = new FormData();
            formData.append("audio", audioBlob, "recording.webm");

            const response = await fetch(
              "http://localhost:5001/api/transcribe",
              {
                method: "POST",
                body: formData,
              }
            );

            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success && data.transcript) {
              setInput(data.transcript);
              // Auto-send the transcribed message - mark as voice input so we speak the response
              setTimeout(() => {
                handleSendMessage(data.transcript, true); // true = voice input, speak response
              }, 100);
            } else {
              console.error("Transcription failed:", data.error);
              alert(
                "Failed to transcribe audio: " + (data.error || "Unknown error")
              );
              setInput("");
            }
          } catch (error) {
            console.error("Error transcribing audio:", error);
            alert(
              "Failed to transcribe audio: " +
                (error.message || "Please try again.")
            );
            setInput("");
          } finally {
            // Clean up
            audioChunksRef.current = [];
            if (currentStream) {
              currentStream.getTracks().forEach((track) => track.stop());
              audioStreamRef.current = null;
            }
          }
        };

        mediaRecorder.onerror = (event) => {
          console.error("MediaRecorder error:", event);
          setIsListening(false);
          setInput("");
          alert("Error recording audio. Please try again.");
          if (audioStreamRef.current) {
            audioStreamRef.current.getTracks().forEach((track) => track.stop());
            audioStreamRef.current = null;
          }
        };

        mediaRecorderRef.current = mediaRecorder;
        audioStreamRef.current = stream; // Store stream separately

        setInput("🎤 Listening... Speak now");
        setIsListening(true);

        // Start recording with timeslice (collect data every second)
        try {
          mediaRecorder.start(1000);
        } catch (startError) {
          console.error("Error starting MediaRecorder:", startError);
          setIsListening(false);
          setInput("");
          if (audioStreamRef.current) {
            audioStreamRef.current.getTracks().forEach((track) => track.stop());
            audioStreamRef.current = null;
          }
          alert("Error starting recording. Please try again.");
        }
      } catch (error) {
        console.error("Unexpected error:", error);
        alert(
          "An unexpected error occurred: " +
            (error.message || "Please try again.")
        );
        setInput("");
        setIsListening(false);
      }
    }
  };

  return (
    <div className="chat-agent">
      <div className="chat-header">
        <div className="chat-header-content">
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
              flex: 1,
            }}>
            <div className="chat-agent-avatar">🔧</div>
            <div>
              <h3>Estimate Specialist</h3>
              <p className="chat-status">Online • Ready to help</p>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {isSpeaking && (
              <div className="speaking-indicator" title="Speaking...">
                <span className="pulse-dot"></span>
              </div>
            )}
            <button
              type="button"
              className={`chat-speech-toggle ${
                speechEnabled ? "enabled" : "disabled"
              }`}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const wasEnabled = speechEnabled;
                toggleSpeech();
                // Test TTS immediately on click (user interaction) if we just enabled it
                if (!wasEnabled) {
                  console.log("🧪 Testing TTS on toggle (enabling speech)...");
                  setTimeout(() => {
                    speakText(
                      "Speech is now enabled. You should hear this message."
                    );
                  }, 300);
                }
              }}
              title={
                speechEnabled
                  ? "Disable speech (currently ON)"
                  : "Enable speech (currently OFF) - Click to test"
              }
              aria-label={speechEnabled ? "Disable speech" : "Enable speech"}>
              {speechEnabled ? (
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              ) : (
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2">
                  <path d="M8 5v14l11-7z" />
                  <line x1="2" y1="2" x2="22" y2="22" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
            {/* Show zoomed images if this is the latest assistant message and we have zoomed images */}
            {msg.role === "assistant" &&
              idx === messages.length - 1 &&
              zoomedImages.length > 0 && (
                <div className="zoomed-boxes-container">
                  <h4>
                    Zoomed Views -{" "}
                    {zoomedImages.length === 1 ? "Damage Area" : "Damage Areas"}
                  </h4>
                  <div
                    className={`zoomed-boxes-grid ${
                      zoomedImages.length > 1 ? "scrollable" : ""
                    }`}>
                    {zoomedImages.map((zoomedData, imgIdx) => (
                      <div key={imgIdx} className="zoomed-box-item">
                        <div className="zoomed-box-header">
                          <span>Damage Area #{zoomedData.boxIndex + 1}</span>
                        </div>
                        <img
                          src={zoomedData.image}
                          alt={`Zoomed damage area ${zoomedData.boxIndex + 1}`}
                          className="zoomed-box-image"
                        />
                        <div className="zoomed-box-info">
                          <p>
                            <strong>Damage Type:</strong>{" "}
                            {zoomedData.info.damage_class}
                          </p>
                          <p>
                            <strong>Severity:</strong>{" "}
                            {zoomedData.info.severity_class}
                          </p>
                          <p>
                            <strong>Cost:</strong> $
                            {zoomedData.info.cost.toLocaleString(undefined, {
                              minimumFractionDigits: 2,
                              maximumFractionDigits: 2,
                            })}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
          </div>
        ))}
        {isLoading && (
          <div className="chat-message assistant">
            <div className="message-content">
              <span className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSend}>
        {isSupported && (
          <button
            type="button"
            className={`chat-voice-btn ${isListening ? "listening" : ""}`}
            onClick={handleVoiceClick}
            disabled={isLoading}
            aria-label={isListening ? "Stop listening" : "Start voice input"}>
            {isListening ? (
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            )}
          </button>
        )}
        <input
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            isSupported
              ? "Type or speak your question..."
              : "Ask about your estimate..."
          }
          disabled={isLoading || isListening}
        />
        <button
          type="submit"
          className="chat-send-btn"
          disabled={!input.trim() || isLoading || isListening}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatAgent;
