#!/bin/bash

# Script to generate password file for nginx basic auth
# Usage: ./generate_passwords.sh

HTPASSWD_FILE="./nginx/.htpasswd"

echo "🔐 Setting up nginx Basic Authentication for Experiment Server"
echo "============================================================="

# Create nginx directory if it doesn't exist
mkdir -p ./nginx

# Function to add user
add_user() {
    local username=$1
    echo
    echo "Adding user: $username"
    
    # Check if htpasswd command exists
    if command -v htpasswd >/dev/null 2>&1; then
        # Use system htpasswd
        htpasswd -c "$HTPASSWD_FILE" "$username"
    elif command -v docker >/dev/null 2>&1; then
        # Use docker with httpd image
        echo "Enter password for $username:"
        read -s password
        echo "$username:$(docker run --rm httpd:alpine htpasswd -nbB "$username" "$password" | cut -d: -f2)" > "$HTPASSWD_FILE"
        echo "Password set for $username"
    else
        echo "❌ Error: Neither htpasswd nor docker is available"
        echo "Please install apache2-utils (htpasswd) or docker"
        exit 1
    fi
}

# Default users for lab access
echo "Setting up default lab users..."
echo "You can add more users later by running this script again"
echo

# Researcher account
add_user "researcher"

# Admin account (optional)
echo
read -p "Add admin account? (y/n): " add_admin
if [[ $add_admin =~ ^[Yy]$ ]]; then
    add_user "admin"
fi

# Lab member accounts (optional)
echo
read -p "Add additional lab member accounts? (y/n): " add_members
if [[ $add_members =~ ^[Yy]$ ]]; then
    while true; do
        echo
        read -p "Enter username (or 'done' to finish): " username
        if [[ $username == "done" ]]; then
            break
        fi
        if [[ -n $username ]]; then
            # For additional users, append to file instead of overwriting
            if command -v htpasswd >/dev/null 2>&1; then
                htpasswd "$HTPASSWD_FILE" "$username"
            else
                echo "Enter password for $username:"
                read -s password
                echo "$username:$(docker run --rm httpd:alpine htpasswd -nbB "$username" "$password" | cut -d: -f2)" >> "$HTPASSWD_FILE"
                echo "Password set for $username"
            fi
        fi
    done
fi

echo
echo "✅ Password file created: $HTPASSWD_FILE"
echo "🔒 Authentication is now configured for:"
echo "   - Main application access"
echo "   - Experiment upload/edit functions"
echo
echo "💡 To add more users later, run: htpasswd $HTPASSWD_FILE <username>"
echo "💡 To remove users, edit $HTPASSWD_FILE manually"
echo
echo "Next steps:"
echo "1. Review nginx/nginx.conf configuration"
echo "2. Run: docker compose up -d"
echo "3. Access your experiment server with the credentials you just created"
