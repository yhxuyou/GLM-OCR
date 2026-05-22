#!/usr/bin/env bash
# ============================================================
# SSL/TLS 证书生成脚本
# ============================================================
# 两种模式:
#   1. 自签名证书 (开发/测试环境)
#   2. Let's Encrypt 证书 (生产环境, 需域名和公网 IP)
# ============================================================
set -euo pipefail

DOMAIN="${1:-glmocr.local}"
MODE="${2:-selfsigned}"
SSL_DIR="$(cd "$(dirname "$0")" && pwd)/ssl"

mkdir -p "$SSL_DIR"

case "$MODE" in
    selfsigned)
        echo "生成自签名 SSL 证书..."
        echo "  域名: $DOMAIN"
        echo "  输出: $SSL_DIR"
        echo ""

        openssl req -x509 -nodes -days 3650 -newkey rsa:4096 \
            -keyout "$SSL_DIR/privkey.pem" \
            -out "$SSL_DIR/fullchain.pem" \
            -subj "/CN=$DOMAIN" \
            -addext "subjectAltName=DNS:$DOMAIN,DNS:localhost,IP:127.0.0.1" \
            2>/dev/null

        echo "自签名证书生成完成!"
        echo "  Certificate: $SSL_DIR/fullchain.pem"
        echo "  Private Key: $SSL_DIR/privkey.pem"
        ;;

    letsencrypt)
        if ! command -v certbot &> /dev/null; then
            echo "安装 certbot..."
            apt-get update && apt-get install -y certbot
        fi

        echo "申请 Let's Encrypt 证书..."
        echo "  域名: $DOMAIN"
        echo "  确保域名 DNS 已指向本机 IP"

        certbot certonly --standalone \
            -d "$DOMAIN" \
            --non-interactive \
            --agree-tos \
            --email "admin@$DOMAIN" \
            --http-01-port 8888

        # 复制到 Nginx SSL 目录
        cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/"
        cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/"

        echo "Let's Encrypt 证书生成完成!"
        echo "  证书每 90 天过期, 建议配置自动续期:"
        echo "  crontab -e"
        echo "  添加: 0 3 * * * certbot renew --quiet && docker exec nginx nginx -s reload"
        ;;

    *)
        echo "未知模式: $MODE"
        echo "使用方式: $0 <域名> <selfsigned|letsencrypt>"
        exit 1
        ;;
esac

# 设置权限
chmod 600 "$SSL_DIR/privkey.pem"
chmod 644 "$SSL_DIR/fullchain.pem"

echo ""
echo "SSL 证书就绪: $SSL_DIR"