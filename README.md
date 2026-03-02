# AI Stock Analysis with SEC RAG

An intelligent stock analysis platform that combines Retrieval-Augmented Generation (RAG) with SEC filings to provide comprehensive stock insights powered by AI.

## Features

- **SEC RAG Embeddings**: Retrieves and analyzes SEC filings using embedding-based search
- **Discord Bot Integration**: Real-time stock analysis alerts through Discord
- **Streamlit Dashboard**: Interactive web interface for exploring stock data
- **Automated Scheduling**: Cron-based automation for periodic analysis tasks
- **Multi-Source Analysis**: Integrates multiple data sources for comprehensive insights

## Project Structure

- `main.py` - Main application entry point
- `streamlit_app.py` - Streamlit dashboard interface
- `discord_bot.py` - Discord bot integration for real-time alerts
- `rag_embeddings.py` - RAG system with embedding-based retrieval
- `sec_filings.py` - SEC filing data extraction and processing
- `constants.py` - Application constants and configuration
- `setup_cron.sh` - Cron job setup script for automation
- `requirements.txt` - Python dependencies

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Update `constants.py` with your configuration settings:
- API keys and credentials
- Database connections
- Discord webhook URLs
- RAG model parameters

## Usage

### Web Dashboard
```bash
streamlit run streamlit_app.py
```

### Main Application
```bash
python main.py
```

### Discord Bot
Configure your Discord webhook in `constants.py`, then run:
```bash
python discord_bot.py
```

### Automated Tasks
Set up cron jobs with:
```bash
bash setup_cron.sh
```

## Data Sources

- **SEC Filings**: 10-K, 10-Q, 8-K forms via SEC EDGAR
- **Stock Data**: Real-time and historical market data
- **NLP Analysis**: AI-powered sentiment and fundamental analysis

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- Streamlit for web interface
- Discord.py for bot integration
- RAG frameworks for semantic search
- SEC Edgar API tools

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For questions or issues, please open an issue in the repository.
