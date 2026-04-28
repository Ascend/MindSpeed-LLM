#!/bin/bash
set -e

echo "=========================================="
echo "Configuring yum repository..."
echo "=========================================="

if [ -f /etc/os-release ]; then
    source /etc/os-release
else
    echo "ERROR: Cannot find /etc/os-release"
    exit 1
fi

cat /etc/os-release

if [[ "$VERSION" =~ SP ]]; then
    SP_VERSION=$(echo "$VERSION" | grep -oP 'SP\d+')
    REPO_DIR="openEuler-${VERSION_ID}-LTS-${SP_VERSION}"
else
    REPO_DIR="openEuler-${VERSION_ID}-LTS"
fi

echo "Detected version: $VERSION"
echo "Repository path: $REPO_DIR"

mkdir -p /etc/yum.repos.d/backup
mv /etc/yum.repos.d/*.repo /etc/yum.repos.d/backup/ 2>/dev/null || true

cat > /etc/yum.repos.d/openEuler.repo <<EOF
[openEuler-everything]
name=openEuler-${VERSION_ID} everything
baseurl=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/everything/\$basearch/
enabled=1
gpgcheck=1
gpgkey=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/everything/\$basearch/RPM-GPG-KEY-openEuler

[openEuler-update]
name=openEuler-${VERSION_ID} update
baseurl=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/update/\$basearch/
enabled=1
gpgcheck=1
gpgkey=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/update/\$basearch/RPM-GPG-KEY-openEuler

[openEuler-debuginfo]
name=openEuler-${VERSION_ID} debuginfo
baseurl=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/debuginfo/\$basearch/
enabled=0
gpgcheck=1
gpgkey=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/debuginfo/\$basearch/RPM-GPG-KEY-openEuler

[openEuler-EPOL]
name=openEuler-${VERSION_ID} EPOL
baseurl=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/EPOL/main/\$basearch/
enabled=1
gpgcheck=1
gpgkey=https://repo.huaweicloud.com/openeuler/${REPO_DIR}/EPOL/main/\$basearch/RPM-GPG-KEY-openEuler
EOF

echo "Repo file generated: /etc/yum.repos.d/openEuler.repo"
echo "Original files backed up to: /etc/yum.repos.d/backup/"

yum clean all

MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    echo ">>> yum makecache attempt $i of $MAX_RETRIES"
    if yum makecache; then
        echo "=========================================="
        echo "Yum repository configured successfully!"
        echo "=========================================="
        exit 0
    fi
    echo ">>> yum makecache failed, retrying in 5 seconds..."
    sleep 5
done

echo "WARNING: yum makecache failed after $MAX_RETRIES attempts"
echo "Repository is configured but metadata cache could not be downloaded."
echo "This may be a temporary network issue. Packages will be downloaded on demand."
echo "=========================================="
