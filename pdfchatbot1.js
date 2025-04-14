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

function splitText(text) {
  const chunks = [];
  for (let i = 0; i < text.length; i += CHUNK_SIZE - CHUNK_OVERLAP) {
    chunks.push(text.slice(i, i + CHUNK_SIZE));
  }
  return chunks;
}

async function loadAndChunkPDF(filePath, label) {
  const buffer = fs.readFileSync(filePath);
  const data = await pdfParse(buffer);
  const rawChunks = splitText(data.text);
  return rawChunks.map(chunk => `${label}: ${chunk}`);
}

async function getEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  return response.data[0].embedding;
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

async function initStore() {
  const userChunks = await loadAndChunkPDF("user_manual.pdf", "USER");
  const operatorChunks = await loadAndChunkPDF("operator_manual.pdf", "OPERATOR");

 /* //  Add these logs
  console.log(" User manual chunks:", userChunks.length);
  console.log(" Operator manual chunks:", operatorChunks.length);
  console.log(" Sample USER chunk:", userChunks[0]?.slice(0, 100));
  console.log(" Sample OPERATOR chunk:", operatorChunks[0]?.slice(0, 100));*/

  const allChunks = [...userChunks, ...operatorChunks];

  for (const chunk of allChunks) {
    const embedding = await getEmbedding(chunk);
    CHUNKS.push(chunk);
    EMBEDDINGS.push(embedding);
  }

  console.log(`Loaded ${CHUNKS.length} chunks into memory.`);
}

async function searchRelevantChunk(query) {
  const queryEmbedding = await getEmbedding(query);
  const scores = EMBEDDINGS.map((embedding, index) => ({
    index,
    score: cosineSimilarity(queryEmbedding, embedding),
  }));

  scores.sort((a, b) => b.score - a.score);

  const SIMILARITY_THRESHOLD = 0.6;
  if (scores[0].score < SIMILARITY_THRESHOLD) {
    return null;
  }

  const topChunks = scores.slice(0, 3).map(score => {
    const chunk = CHUNKS[score.index];
    console.log(chalk.gray(`\n Matched Chunk (${score.score.toFixed(3)}):\n${chunk}\n`));
  return chunk;
}).join("\n\n");

  return topChunks;
}

async function askChat(chunk, query) {
  const prompt = `
You are a precise and factual assistant. 
Use only the information from the provided manuals (below) to answer the user's question.
Do not assume or make up any details.
Maintain original units, numbers, and terminology.

If the answer is not in the provided content, simply respond:
"I'm sorry, the answer is not available in the provided content."


Manual Excerpts:
${chunk}


Question: ${query}
Answer:`;

  const completion = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: prompt }],
    temperature: 0.3,
  });

  return completion.choices[0].message.content.trim();
}

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
        return promptUser(); // Skip blank input
      }

      if (question.trim().toLowerCase() === "exit") {
        console.log(chalk.green('\n Consultant: Thank you! Have a productive day!\n'));
        rl.close();
        process.exit(0);
      }

      const chunk = await searchRelevantChunk(question);

      if (!chunk) {
        console.log(chalk.yellow("\n Consultant: Please ask a valid or related question based on the manuals."));
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
