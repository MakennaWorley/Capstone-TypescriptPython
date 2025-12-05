FROM node:22-alpine AS deps
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install --frozen-lockfile --build-if-needed

FROM node:22-alpine AS stage-1
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend ./
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]