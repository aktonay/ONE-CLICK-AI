// ================================================================
// ONE CLICK AI SPARK - Custom Documentation JavaScript
// Interactive Animations & Enhanced User Experience
// ================================================================

(function() {
    'use strict';

    // ============ WAIT FOR DOM LOAD ============
    document.addEventListener('DOMContentLoaded', function() {
        initAnimations();
        initScrollEffects();
        initCodeCopyButtons();
        initDynamicLinks();
        initParticles();
        initTypingEffect();
    });

    // ============ INITIALIZE ANIMATIONS ============
    function initAnimations() {
        // Fade in elements on scroll
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1
        });

        document.querySelectorAll('.section, h2, h3, table, pre').forEach(el => {
            observer.observe(el);
        });
    }

    // ============ SCROLL EFFECTS ============
    function initScrollEffects() {
        let ticking = false;

        window.addEventListener('scroll', () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    handleScroll();
                    ticking = false;
                });
                ticking = true;
            }
        });

        function handleScroll() {
            const scrolled = window.pageYOffset;
            
            // Parallax effect for headers
            document.querySelectorAll('h1').forEach(h1 => {
                const speed = 0.5;
                h1.style.transform = `translateY(${scrolled * speed}px)`;
            });

            // Add shadow to top bar when scrolled
            const topBar = document.querySelector('.wy-nav-top');
            if (topBar) {
                if (scrolled > 50) {
                    topBar.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.3)';
                } else {
                    topBar.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.1)';
                }
            }
        }
    }

    // ============ CODE COPY BUTTONS ============
    function initCodeCopyButtons() {
        document.querySelectorAll('pre').forEach(pre => {
            // Create copy button
            const button = document.createElement('button');
            button.className = 'copy-button';
            button.innerHTML = '📋 Copy';
            button.style.cssText = `
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.5rem 1rem;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;
                transition: all 0.3s;
                z-index: 10;
            `;

            button.addEventListener('mouseenter', () => {
                button.style.transform = 'scale(1.05)';
                button.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.5)';
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'scale(1)';
                button.style.boxShadow = 'none';
            });

            button.addEventListener('click', async () => {
                const code = pre.querySelector('code');
                const text = code ? code.textContent : pre.textContent;
                
                try {
                    await navigator.clipboard.writeText(text);
                    button.innerHTML = '✅ Copied!';
                    button.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                    
                    setTimeout(() => {
                        button.innerHTML = '📋 Copy';
                        button.style.background = 'linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%)';
                    }, 2000);
                } catch (err) {
                    button.innerHTML = '❌ Failed';
                    setTimeout(() => {
                        button.innerHTML = '📋 Copy';
                    }, 2000);
                }
            });

            pre.style.position = 'relative';
            pre.appendChild(button);
        });
    }

    // ============ DYNAMIC LINKS WITH ICONS ============
    function initDynamicLinks() {
        // Add icons to external links
        document.querySelectorAll('a[href^="http"]').forEach(link => {
            if (!link.querySelector('.link-icon')) {
                const icon = document.createElement('span');
                icon.className = 'link-icon';
                icon.innerHTML = ' 🔗';
                icon.style.cssText = `
                    opacity: 0.6;
                    font-size: 0.85em;
                    transition: all 0.3s;
                `;
                link.appendChild(icon);

                link.addEventListener('mouseenter', () => {
                    icon.style.opacity = '1';
                    icon.style.transform = 'translateX(3px)';
                });

                link.addEventListener('mouseleave', () => {
                    icon.style.opacity = '0.6';
                    icon.style.transform = 'translateX(0)';
                });
            }
        });

        // Add GitHub icon to GitHub links
        document.querySelectorAll('a[href*="github.com"]').forEach(link => {
            const icon = link.querySelector('.link-icon');
            if (icon) {
                icon.innerHTML = ' ⭐';
            }
        });

        // Add PyPI icon to PyPI links
        document.querySelectorAll('a[href*="pypi.org"]').forEach(link => {
            const icon = link.querySelector('.link-icon');
            if (icon) {
                icon.innerHTML = ' 📦';
            }
        });
    }

    // ============ PARTICLE BACKGROUND ============
    function initParticles() {
        const canvas = document.createElement('canvas');
        canvas.id = 'particles';
        canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.3;
        `;
        document.body.prepend(canvas);

        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const particles = [];
        const particleCount = 50;

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.radius = Math.random() * 2 + 1;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
                ctx.fill();
            }
        }

        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach(particle => {
                particle.update();
                particle.draw();
            });

            // Draw connections
            particles.forEach((p1, i) => {
                particles.slice(i + 1).forEach(p2 => {
                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 150) {
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(0, 212, 255, ${0.2 * (1 - distance / 150)})`;
                        ctx.stroke();
                    }
                });
            });

            requestAnimationFrame(animate);
        }

        animate();

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    }

    // ============ TYPING EFFECT FOR H1 ============
    function initTypingEffect() {
        const h1Elements = document.querySelectorAll('h1');
        
        h1Elements.forEach(h1 => {
            const text = h1.textContent;
            h1.textContent = '';
            h1.style.opacity = '1';
            
            let index = 0;
            const cursor = document.createElement('span');
            cursor.textContent = '|';
            cursor.style.cssText = `
                animation: blink 1s infinite;
                color: #00d4ff;
            `;
            
            // Add blink animation
            if (!document.getElementById('blink-style')) {
                const style = document.createElement('style');
                style.id = 'blink-style';
                style.textContent = `
                    @keyframes blink {
                        0%, 50% { opacity: 1; }
                        51%, 100% { opacity: 0; }
                    }
                `;
                document.head.appendChild(style);
            }

            function type() {
                if (index < text.length) {
                    h1.textContent = text.substring(0, index + 1);
                    h1.appendChild(cursor);
                    index++;
                    setTimeout(type, 50);
                } else {
                    setTimeout(() => cursor.remove(), 1000);
                }
            }

            // Start typing after a short delay
            setTimeout(type, 500);
        });
    }

    // ============ SMOOTH SCROLL FOR ANCHOR LINKS ============
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // ============ PROGRESS BAR ============
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 50%, #f59e0b 100%);
        z-index: 9999;
        transition: width 0.1s;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    `;
    document.body.prepend(progressBar);

    window.addEventListener('scroll', () => {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        progressBar.style.width = scrolled + '%';
    });

    // ============ BACK TO TOP BUTTON ============
    const backToTop = document.createElement('button');
    backToTop.innerHTML = '⬆️';
    backToTop.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        color: white;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        opacity: 0;
        transition: all 0.3s;
        z-index: 1000;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
    `;

    document.body.appendChild(backToTop);

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.style.opacity = '1';
        } else {
            backToTop.style.opacity = '0';
        }
    });

    backToTop.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    backToTop.addEventListener('mouseenter', () => {
        backToTop.style.transform = 'scale(1.1) translateY(-5px)';
        backToTop.style.boxShadow = '0 8px 30px rgba(0, 212, 255, 0.5)';
    });

    backToTop.addEventListener('mouseleave', () => {
        backToTop.style.transform = 'scale(1) translateY(0)';
        backToTop.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.3)';
    });

})();
