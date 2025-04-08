const fs = require("fs");
const pdfParse = require("pdf-parse");
const readline = require("readline");
const dotenv = require("dotenv");
const OpenAI = require("openai");
const chalk = require("chalk"); 

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;
const CHUNKS = [];
const EMBEDDINGS = [];

// Split text into chunks with overlap
function splitText(text) {
  const chunks = [];
  for (let i = 0; i < text.length; i += CHUNK_SIZE - CHUNK_OVERLAP) {
    chunks.push(text.slice(i, i + CHUNK_SIZE));
  }
  return chunks;
}

// Load and parse PDF
async function loadAndChunkPDF(filePath) {
  const buffer = fs.readFileSync(filePath);
  const data = await pdfParse(buffer);
  return splitText(data.text);
}

// Get embeddings
async function getEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  return response.data[0].embedding;
}

// Calculate cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

// Create vector store from PDF chunks
async function initStore() {

  const userChunks = await loadAndChunkPDF("user_manual.pdf");
  const operatorChunks = await loadAndChunkPDF("operator_manual.pdf");
  const allChunks = [...userChunks, ...operatorChunks];

  for (const chunk of allChunks) {
    const embedding = await getEmbedding(chunk);
    CHUNKS.push(chunk);
    EMBEDDINGS.push(embedding);
  }

 console.log(`Loaded ${CHUNKS.length} chunks into memory.`);

}

// Find best matching chunk
async function searchRelevantChunk(query) {
  const queryEmbedding = await getEmbedding(query);
  const scores = EMBEDDINGS.map((embedding, index) => ({
    index,
    score: cosineSimilarity(queryEmbedding, embedding),
  }));

  scores.sort((a, b) => b.score - a.score);
  const topMatch = scores[0];

  // Set a threshold for relevance (tweak as needed)
  const SIMILARITY_THRESHOLD = 0.85;

  if (topMatch.score < SIMILARITY_THRESHOLD) {
    return null; // Not relevant
  }

  // Get top 3 chunks and join them
  const topChunks = scores.slice(0, 3).map(score => CHUNKS[score.index]).join("\n\n");

  return topChunks;
}

// Ask GPT using context from PDF
async function askChat(chunk, query) {
  const prompt = `You are a precise and factual assistant. 
Use only the information from the provided manuals (PDF content below) to answer the question. 
Do not assume or fabricate any details. 
Maintain original units, numbers, and terminology from the manuals.
If the answer is not explicitly stated in the content, reply with:

"I'm sorry, the answer is not available in the provided content."\n\nContent:\n${chunk}\n\nQuestion: ${query}\nAnswer:`;

  const completion = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: prompt }],
    temperature: 0.3,
  });

  return completion.choices[0].message.content.trim();
}

// Chat loop
async function runChat() {
  await initStore();

  console.log(chalk.bold.green('\n Welcome to the Manual Assistant!'));
  console.log(chalk.green(' Ask questions related to the manuals. Type "exit" to quit.\n'));

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const promptUser = () => {
    rl.question(chalk.blueBright('\nYou: '), async (question) => {
      if (!question.trim()) {
        return promptUser(); // Skip empty input
      }

      if (question.trim().toLowerCase() === "exit") {
        console.log(chalk.green('\n Consultant: Thank you! Have a productive day!\n'));
        rl.close();
        process.exit(0);
      }

      const chunk = await searchRelevantChunk(question);

      if (!chunk) {
        console.log(chalk.yellow("\n  Consultant: Please ask a valid or related question based on the manuals."));
        return promptUser();
      }

      const answer = await askChat(chunk, question);
      console.log(chalk.cyanBright("\n Consultant:"), chalk.whiteBright(answer));
      promptUser();
    });
  };

  promptUser();
}

runChat();
