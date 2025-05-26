#!/bin/bash

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install required packages
echo "Installing required packages..."
sudo apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv git nginx

# Install Docker
echo "Installing Docker..."
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Set up Python environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pull Gemma model
echo "Pulling Gemma model (this may take some time)..."
ollama pull gemma3:4b

# Set up services
echo "Setting up services..."
sudo cp ollama.service /etc/systemd/system/
sudo cp gemma-api.service /etc/systemd/system/
sudo cp nginx-gemma-api /etc/nginx/sites-available/gemma-api
sudo ln -s /etc/nginx/sites-available/gemma-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Update Nginx config with the correct hostname
PUBLIC_DNS=$(curl -s http://169.254.169.254/latest/meta-data/public-hostname)
sudo sed -i "s/server_name _;/server_name $PUBLIC_DNS;/" /etc/nginx/sites-available/gemma-api

# Start services
echo "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl enable gemma-api
sudo systemctl start gemma-api
sudo systemctl restart nginx

echo "Setup complete! Your API should be available at http://$PUBLIC_DNS"
echo "To check the status of your services:"
echo "  - Ollama: sudo systemctl status ollama"
echo "  - Gemma API: sudo systemctl status gemma-api"
echo "  - Nginx: sudo systemctl status nginx"
