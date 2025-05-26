# AWS EC2 Deployment Package for Gemma API

This package contains all the files and guides needed to deploy your FastAPI Gemma-3 application to AWS EC2.

## Supported Ubuntu Versions

This deployment package supports both:
- **Ubuntu 22.04 LTS** (default setup)
- **Ubuntu 24.04 LTS** (with additional setup script and guide)

## Directory Structure

I've organized the deployment files into a structured layout for easier management:

```
gemma_api_deployment/
│
├── guides/                      # Detailed guides for deployment and maintenance
│   ├── DEPLOYMENT_GUIDE.md      # Comprehensive deployment guide
│   ├── EC2_SETUP_GUIDE.md       # Step-by-step EC2 instance setup
│   ├── MAINTENANCE_GUIDE.md     # Guide for maintaining your deployment
│   ├── PUTTY_GUIDE.md           # Guide for connecting with PuTTY
│   ├── TROUBLESHOOTING.md       # Solutions for common issues
│   ├── UBUNTU24_GUIDE.md        # Specific guide for Ubuntu 24.04 deployment
│   └── WINSCP_GUIDE.md          # Guide for file transfers with WinSCP
│
└── deployment_files/            # Files needed on the EC2 instance
    ├── gemma-api.service        # Systemd service for FastAPI app
    ├── nginx-gemma-api          # Nginx configuration
    ├── ollama.service           # Systemd service for Ollama
    ├── setup.sh                 # Automated setup script for Ubuntu 22.04
    └── setup_ubuntu24.sh        # Automated setup script for Ubuntu 24.04
```

## How to Organize the Files

To organize the files into this structure, follow these steps:

1. Create the directory structure:
   ```
   mkdir -p gemma_api_deployment\guides gemma_api_deployment\deployment_files
   ```

2. Move the guide files to the guides directory:
   ```
   copy DEPLOYMENT_GUIDE.md gemma_api_deployment\guides\
   copy TROUBLESHOOTING.md gemma_api_deployment\guides\
   copy EC2_SETUP_GUIDE.md gemma_api_deployment\guides\
   copy WINSCP_GUIDE.md gemma_api_deployment\guides\
   copy PUTTY_GUIDE.md gemma_api_deployment\guides\
   copy MAINTENANCE_GUIDE.md gemma_api_deployment\guides\
   copy UBUNTU24_GUIDE.md gemma_api_deployment\guides\
   ```

3. Move the deployment files to the deployment_files directory:
   ```
   copy gemma-api.service gemma_api_deployment\deployment_files\
   copy nginx-gemma-api gemma_api_deployment\deployment_files\
   copy ollama.service gemma_api_deployment\deployment_files\
   copy setup.sh gemma_api_deployment\deployment_files\
   copy setup_ubuntu24.sh gemma_api_deployment\deployment_files\
   ```

4. Copy this README to the main directory:
   ```
   copy README.md gemma_api_deployment\
   ```

## How to Use This Package

1. Start with `guides/EC2_SETUP_GUIDE.md` to create your EC2 instance
2. Use `guides/PUTTY_GUIDE.md` to connect to your instance
3. Follow `guides/WINSCP_GUIDE.md` to upload files
4. Complete the process with `guides/DEPLOYMENT_GUIDE.md`
5. If you're using Ubuntu 24.04, refer to `guides/UBUNTU24_GUIDE.md` for specific instructions
6. If you encounter issues, refer to `guides/TROUBLESHOOTING.md`
7. For ongoing maintenance, follow `guides/MAINTENANCE_GUIDE.md`

## Deployment Process Overview

1. Create an EC2 instance in AWS (t3.large or better recommended)
   - Choose Ubuntu 22.04 LTS or Ubuntu 24.04 LTS
2. Connect to your instance using PuTTY
3. Upload the files from `deployment_files/` using WinSCP
4. Run the appropriate setup script:
   - For Ubuntu 22.04: `./setup.sh`
   - For Ubuntu 24.04: `./setup_ubuntu24.sh`
5. Test your API endpoints
6. Set up regular maintenance and backups

## Files to Upload to EC2

When setting up your EC2 instance, you need to upload:
1. Your `main.py` file
2. All files in the `deployment_files/` directory
3. Your `requirements.txt` file (if not already included)

## Important Notes

- The setup process will take some time, especially downloading the Gemma-3 model
- Make sure your EC2 instance has enough resources (CPU, RAM, and disk space)
- Keep your .pem key file secure - you cannot download it again from AWS
- Regularly update your system and create backups
