## Custom Scripting For Using QwenTTS

This is basically a `bash` and Python script that makes it easier to interact with Qwen's text to speech model for creating custom voices.

You give it a video or audio of someone talking, it gives you back a wave in the proper audio format for Qwen, then Whisper parses the audio for words, and you feed the audio file and the transcript that was generated in order to output speech of your own creation!
