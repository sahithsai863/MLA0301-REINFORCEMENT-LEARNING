// ============================================================
// TIME RIFT
// Adaptive Reinforcement Learning Game
// ============================================================


// ============================================================
// CANVAS
// ============================================================

const canvas =
    document.getElementById("canvas");

const ctx =
    canvas.getContext("2d");


let W = window.innerWidth;

let H = window.innerHeight;


function resize() {

    W = canvas.width =
        window.innerWidth;

    H = canvas.height =
        window.innerHeight;

}

window.addEventListener(
    "resize",
    resize
);

resize();


// ============================================================
// GAME STATE
// ============================================================

let gameRunning = false;

let paused = false;

let lastTime = 0;

let elapsed = 0;

let timeLeft = 120;

let score = 0;

let shards = 0;

let combo = 0;

let health = 3;

let energy = 100;

let level = 1;


// ============================================================
// PLAYER
// ============================================================

const player = {

    x: 0,

    y: 0,

    radius: 16,

    speed: 260,

    dashSpeed: 850,

    dashTime: 0,

    invincible: 0,

    directionX: 0,

    directionY: -1

};


// ============================================================
// KEYBOARD
// ============================================================

const keys = {};

window.addEventListener(
    "keydown",
    e => {

        keys[e.key.toLowerCase()] = true;

        if (
            e.code === "Space"
        ) {

            e.preventDefault();

            keys.space = true;

        }


        if (
            e.key === "Escape"
        ) {

            togglePause();

        }


        if (
            e.key.toLowerCase() === "e"
        ) {

            useRift();

        }

    }
);


window.addEventListener(
    "keyup",
    e => {

        keys[e.key.toLowerCase()] = false;

        if (
            e.code === "Space"
        ) {

            keys.space = false;

        }

    }
);


// ============================================================
// OBJECTS
// ============================================================

let crystals = [];

let enemies = [];

let energyOrbs = [];

let hazards = [];

let portals = [];

let particles = [];


// ============================================================
// GAME DIRECTOR
// ============================================================

let director = {

    aggression: 0.35,

    spawnRate: 2.5,

    rewardRate: 1.0

};


// ============================================================
// Q LEARNING
// ============================================================

const ALPHA = 0.15;

const GAMMA = 0.85;

let qTable = {};


// ============================================================
// GAME DIRECTOR ACTIONS
// ============================================================

const directorActions = [

    "EASY",

    "MORE_ENEMIES",

    "MORE_REWARDS",

    "DANGER_ZONE",

    "ENERGY_DROP"

];


// ============================================================
// Q STATE
// ============================================================

function getRLState() {

    const scoreState =
        score < 500
            ? 0
            : score < 1500
                ? 1
                : 2;


    const healthState =
        health === 1
            ? 0
            : health === 2
                ? 1
                : 2;


    const comboState =
        combo < 2
            ? 0
            : combo < 5
                ? 1
                : 2;


    const energyState =
        energy < 25
            ? 0
            : energy < 60
                ? 1
                : 2;


    return (
        `${scoreState}_` +
        `${healthState}_` +
        `${comboState}_` +
        `${energyState}`
    );

}


// ============================================================
// ENSURE Q STATE
// ============================================================

function ensureRLState(state) {

    if (!qTable[state]) {

        qTable[state] = [
            0,
            0,
            0,
            0,
            0
        ];

    }

}


// ============================================================
// CHOOSE DIRECTOR ACTION
// ============================================================

function chooseDirectorAction() {

    const state =
        getRLState();

    ensureRLState(state);


    const values =
        qTable[state];


    let best = 0;


    for (
        let i = 1;
        i < values.length;
        i++
    ) {

        if (
            values[i] >
            values[best]
        ) {

            best = i;

        }

    }


    return best;

}


// ============================================================
// UPDATE DIRECTOR
// ============================================================

function updateDirector() {

    const action =
        chooseDirectorAction();


    switch (action) {

        case 0:

            director.aggression =
                Math.max(
                    0.2,
                    director.aggression -
                    0.05
                );

            break;


        case 1:

            director.aggression =
                Math.min(
                    0.9,
                    director.aggression +
                    0.08
                );

            spawnEnemy();

            break;


        case 2:

            spawnCrystal();

            spawnCrystal();

            break;


        case 3:

            spawnHazard();

            break;


        case 4:

            spawnEnergy();

            break;

    }

}


// ============================================================
// START GAME
// ============================================================

function startGame() {

    document
        .getElementById(
            "startScreen"
        )
        .classList.add(
            "hidden"
        );


    resetGame();


    gameRunning = true;

    paused = false;


    lastTime =
        performance.now();


    requestAnimationFrame(
        gameLoop
    );

}


// ============================================================
// RESET GAME
// ============================================================

function resetGame() {

    elapsed = 0;

    timeLeft = 120;

    score = 0;

    shards = 0;

    combo = 0;

    health = 3;

    energy = 100;

    level = 1;


    crystals = [];

    enemies = [];

    energyOrbs = [];

    hazards = [];

    portals = [];

    particles = [];


    player.x =
        W / 2;

    player.y =
        H / 2;


    player.dashTime = 0;

    player.invincible = 0;


    director.aggression =
        0.35;


    // Initial world

    for (
        let i = 0;
        i < 7;
        i++
    ) {

        spawnCrystal();

    }


    for (
        let i = 0;
        i < 2;
        i++
    ) {

        spawnEnemy();

    }


    for (
        let i = 0;
        i < 2;
        i++
    ) {

        spawnEnergy();

    }


    createPortal();


    updateHUD();

}


// ============================================================
// RANDOM POSITION
// ============================================================

function randomPosition() {

    const margin = 80;


    return {

        x:
            margin +
            Math.random() *
            (W - margin * 2),

        y:
            110 +
            Math.random() *
            (H - 170)

    };

}


// ============================================================
// SPAWN CRYSTAL
// ============================================================

function spawnCrystal() {

    const p =
        randomPosition();


    crystals.push({

        x: p.x,

        y: p.y,

        radius: 9,

        pulse:
            Math.random() *
            Math.PI * 2

    });

}


// ============================================================
// SPAWN ENERGY
// ============================================================

function spawnEnergy() {

    const p =
        randomPosition();


    energyOrbs.push({

        x: p.x,

        y: p.y,

        radius: 11,

        pulse: 0

    });

}


// ============================================================
// SPAWN ENEMY
// ============================================================

function spawnEnemy() {

    const p =
        randomPosition();


    const angle =
        Math.random() *
        Math.PI * 2;


    enemies.push({

        x: p.x,

        y: p.y,

        radius: 14,

        speed:
            65 +
            director.aggression *
            80,

        phase: angle

    });

}


// ============================================================
// SPAWN HAZARD
// ============================================================

function spawnHazard() {

    const p =
        randomPosition();


    hazards.push({

        x: p.x,

        y: p.y,

        radius: 35,

        life: 8

    });


    showNotification(
        "⚠ DIMENSIONAL INSTABILITY"
    );

}


// ============================================================
// PORTAL
// ============================================================

function createPortal() {

    const p =
        randomPosition();


    portals.push({

        x: p.x,

        y: p.y,

        radius: 24,

        pulse: 0

    });

}


// ============================================================
// RIFT POWER
// ============================================================

function useRift() {

    if (
        energy < 35 ||
        !gameRunning ||
        paused
    )
        return;


    energy -= 35;


    const p =
        randomPosition();


    player.x = p.x;

    player.y = p.y;


    combo += 1;


    score += 25;


    createParticles(
        player.x,
        player.y,
        "#c084fc",
        25
    );


    showNotification(
        "◈ RIFT JUMP"
    );


    updateHUD();

}


// ============================================================
// DASH
// ============================================================

function dash() {

    if (
        energy < 20 ||
        player.dashTime > 0
    )
        return;


    energy -= 20;

    player.dashTime =
        0.22;


    createParticles(
        player.x,
        player.y,
        "#8b5cf6",
        12
    );


    showNotification(
        "✦ DASH"
    );

}


// ============================================================
// UPDATE PLAYER
// ============================================================

function updatePlayer(dt) {

    let dx = 0;

    let dy = 0;


    if (
        keys.w ||
        keys.arrowup
    )
        dy--;


    if (
        keys.s ||
        keys.arrowdown
    )
        dy++;


    if (
        keys.a ||
        keys.arrowleft
    )
        dx--;


    if (
        keys.d ||
        keys.arrowright
    )
        dx++;


    const length =
        Math.hypot(
            dx,
            dy
        );


    if (length > 0) {

        dx /= length;

        dy /= length;


        player.directionX =
            dx;

        player.directionY =
            dy;

    }


    if (
        keys.space
    ) {

        dash();

        keys.space = false;

    }


    const speed =
        player.dashTime > 0
            ? player.dashSpeed
            : player.speed;


    player.x +=
        dx *
        speed *
        dt;


    player.y +=
        dy *
        speed *
        dt;


    player.x =
        Math.max(
            35,
            Math.min(
                W - 35,
                player.x
            )
        );


    player.y =
        Math.max(
            100,
            Math.min(
                H - 35,
                player.y
            )
        );


    if (
        player.dashTime > 0
    ) {

        player.dashTime -= dt;

    }


    if (
        player.invincible > 0
    ) {

        player.invincible -= dt;

    }


    // Passive energy recovery

    energy =
        Math.min(
            100,
            energy +
            dt * 2
        );

}


// ============================================================
// UPDATE ENEMIES
// ============================================================

function updateEnemies(dt) {

    for (
        const enemy of enemies
    ) {

        const dx =
            player.x -
            enemy.x;


        const dy =
            player.y -
            enemy.y;


        const distance =
            Math.hypot(
                dx,
                dy
            );


        if (
            distance > 0
        ) {

            enemy.x +=
                (
                    dx /
                    distance
                ) *
                enemy.speed *
                dt;


            enemy.y +=
                (
                    dy /
                    distance
                ) *
                enemy.speed *
                dt;

        }


        // Collision

        if (
            distance <
            player.radius +
            enemy.radius
        ) {

            damagePlayer(
                1
            );


            // Push enemy away

            enemy.x -=
                dx *
                0.08;


            enemy.y -=
                dy *
                0.08;

        }

    }

}


// ============================================================
// UPDATE HAZARDS
// ============================================================

function updateHazards(dt) {

    for (
        const hazard of hazards
    ) {

        hazard.life -= dt;


        const distance =
            Math.hypot(
                player.x -
                hazard.x,

                player.y -
                hazard.y
            );


        if (
            distance <
            hazard.radius +
            player.radius
        ) {

            damagePlayer(
                1
            );

        }

    }


    hazards =
        hazards.filter(
            h =>
                h.life > 0
        );

}


// ============================================================
// COLLECT OBJECTS
// ============================================================

function collectObjects() {

    // Crystals

    for (
        let i =
            crystals.length - 1;
        i >= 0;
        i--
    ) {

        const crystal =
            crystals[i];


        const distance =
            Math.hypot(
                player.x -
                crystal.x,

                player.y -
                crystal.y
            );


        if (
            distance <
            player.radius +
            crystal.radius +
            5
        ) {

            crystals.splice(
                i,
                1
            );


            shards++;

            combo++;

            score +=
                10 *
                Math.max(
                    1,
                    combo
                );


            energy =
                Math.min(
                    100,
                    energy + 5
                );


            createParticles(
                crystal.x,
                crystal.y,
                "#d8b4fe",
                18
            );


            showNotification(
                `+${10 * Math.max(1, combo)} SHARDS`
            );

        }

    }


    // Energy

    for (
        let i =
            energyOrbs.length - 1;
        i >= 0;
        i--
    ) {

        const orb =
            energyOrbs[i];


        const distance =
            Math.hypot(
                player.x -
                orb.x,

                player.y -
                orb.y
            );


        if (
            distance <
            player.radius +
            orb.radius
        ) {

            energyOrbs.splice(
                i,
                1
            );


            energy =
                Math.min(
                    100,
                    energy + 30
                );


            score += 20;


            createParticles(
                orb.x,
                orb.y,
                "#67e8f9",
                20
            );


            showNotification(
                "⚡ ENERGY +30"
            );

        }

    }


    // Portal

    for (
        const portal of portals
    ) {

        const distance =
            Math.hypot(
                player.x -
                portal.x,

                player.y -
                portal.y
            );


        if (
            distance <
            player.radius +
            portal.radius
        ) {

            level++;

            score += 100;

            timeLeft += 15;


            createParticles(
                portal.x,
                portal.y,
                "#c084fc",
                35
            );


            portal.x =
                randomPosition().x;

            portal.y =
                randomPosition().y;


            showNotification(
                `◈ RIFT LEVEL ${level}`
            );


            // Increase difficulty

            director.aggression =
                Math.min(
                    .9,
                    director.aggression +
                    .05
                );

        }

    }


    // Combo timeout

    if (
        combo > 0
    ) {

        combo -=
            0.01;

    }

}


// ============================================================
// DAMAGE
// ============================================================

function damagePlayer(amount) {

    if (
        player.invincible > 0
    )
        return;


    health -= amount;

    combo = 0;

    player.invincible =
        1.2;


    score =
        Math.max(
            0,
            score - 50
        );


    createParticles(
        player.x,
        player.y,
        "#ff5577",
        25
    );


    showNotification(
        "♥ DAMAGE"
    );


    if (
        health <= 0
    ) {

        endGame();

    }


    updateHUD();

}


// ============================================================
// DIRECTOR TIMER
// ============================================================

let directorTimer = 0;


function updateDirectorSystem(dt) {

    directorTimer += dt;


    if (
        directorTimer > 3
    ) {

        directorTimer = 0;


        updateDirector();


        // Maintain crystals

        if (
            crystals.length < 5
        ) {

            spawnCrystal();

        }


        // Maintain energy

        if (
            energyOrbs.length < 1
        ) {

            spawnEnergy();

        }


        // Difficulty based on level

        const targetEnemies =
            2 +
            level +
            Math.floor(
                director.aggression * 3
            );


        while (
            enemies.length <
            targetEnemies
        ) {

            spawnEnemy();

        }

    }

}


// ============================================================
// PARTICLES
// ============================================================

function createParticles(
    x,
    y,
    color,
    count
) {

    for (
        let i = 0;
        i < count;
        i++
    ) {

        const angle =
            Math.random() *
            Math.PI *
            2;


        const speed =
            40 +
            Math.random() *
            150;


        particles.push({

            x: x,

            y: y,

            vx:
                Math.cos(angle) *
                speed,

            vy:
                Math.sin(angle) *
                speed,

            life: 0.5 +
                Math.random() *
                0.7,

            color: color,

            size:
                2 +
                Math.random() * 4

        });

    }

}


// ============================================================
// UPDATE PARTICLES
// ============================================================

function updateParticles(dt) {

    for (
        const p of particles
    ) {

        p.x +=
            p.vx *
            dt;


        p.y +=
            p.vy *
            dt;


        p.vx *=
            0.96;


        p.vy *=
            0.96;


        p.life -= dt;

    }


    particles =
        particles.filter(
            p =>
                p.life > 0
        );

}


// ============================================================
// DRAW BACKGROUND
// ============================================================

function drawBackground() {

    ctx.fillStyle =
        "#06040c";


    ctx.fillRect(
        0,
        0,
        W,
        H
    );


    // Stars

    for (
        let i = 0;
        i < 90;
        i++
    ) {

        const x =
            (
                i * 137
            ) % W;


        const y =
            (
                i * 71
            ) % H;


        const alpha =
            0.15 +
            (
                Math.sin(
                    elapsed * 2 +
                    i
                ) + 1
            ) *
            0.12;


        ctx.fillStyle =
            `rgba(200,160,255,${alpha})`;


        ctx.fillRect(
            x,
            y,
            1.5,
            1.5
        );

    }


    // Dimensional rings

    const cx =
        W / 2;


    const cy =
        H / 2;


    for (
        let i = 0;
        i < 5;
        i++
    ) {

        ctx.beginPath();


        ctx.arc(

            cx,
            cy,

            120 +
            i * 90 +
            Math.sin(
                elapsed +
                i
            ) * 20,

            0,
            Math.PI * 2

        );


        ctx.strokeStyle =
            `rgba(150,80,220,${0.06 - i * 0.008})`;


        ctx.lineWidth = 2;

        ctx.stroke();

    }

}


// ============================================================
// DRAW PLAYER
// ============================================================

function drawPlayer() {

    ctx.save();


    // Glow

    ctx.shadowBlur = 25;

    ctx.shadowColor =
        "#a855f7";


    if (
        player.invincible > 0 &&
        Math.floor(
            player.invincible * 10
        ) % 2 === 0
    ) {

        ctx.globalAlpha = 0.3;

    }


    // Outer circle

    ctx.beginPath();

    ctx.arc(
        player.x,
        player.y,
        player.radius + 5,
        0,
        Math.PI * 2
    );

    ctx.fillStyle =
        "#7c3aed";

    ctx.fill();


    // Core

    ctx.beginPath();

    ctx.arc(
        player.x,
        player.y,
        player.radius,
        0,
        Math.PI * 2
    );

    ctx.fillStyle =
        "#d8b4fe";

    ctx.fill();


    // Direction

    ctx.beginPath();

    ctx.moveTo(
        player.x +
        player.directionX *
        25,

        player.y +
        player.directionY *
        25
    );

    ctx.lineTo(
        player.x +
        player.directionX *
        9 -
        player.directionY *
        7,

        player.y +
        player.directionY *
        9 +
        player.directionX *
        7
    );

    ctx.lineTo(
        player.x +
        player.directionX *
        9 +
        player.directionY *
        7,

        player.y +
        player.directionY *
        9 -
        player.directionX *
        7
    );

    ctx.closePath();

    ctx.fillStyle =
        "#ffffff";

    ctx.fill();


    ctx.restore();

}


// ============================================================
// DRAW CRYSTALS
// ============================================================

function drawCrystals() {

    for (
        const crystal of crystals
    ) {

        crystal.pulse +=
            0.05;


        const scale =
            1 +
            Math.sin(
                crystal.pulse
            ) *
            0.2;


        ctx.save();


        ctx.translate(
            crystal.x,
            crystal.y
        );


        ctx.scale(
            scale,
            scale
        );


        ctx.shadowBlur = 20;

        ctx.shadowColor =
            "#d8b4fe";


        ctx.beginPath();

        ctx.moveTo(
            0,
            -12
        );

        ctx.lineTo(
            9,
            0
        );

        ctx.lineTo(
            0,
            12
        );

        ctx.lineTo(
            -9,
            0
        );

        ctx.closePath();


        ctx.fillStyle =
            "#d8b4fe";

        ctx.fill();


        ctx.restore();

    }

}


// ============================================================
// DRAW ENERGY
// ============================================================

function drawEnergy() {

    for (
        const orb of energyOrbs
    ) {

        orb.pulse +=
            0.08;


        const r =
            orb.radius +
            Math.sin(
                orb.pulse
            ) *
            2;


        ctx.save();


        ctx.shadowBlur = 25;

        ctx.shadowColor =
            "#22d3ee";


        ctx.beginPath();

        ctx.arc(
            orb.x,
            orb.y,
            r,
            0,
            Math.PI * 2
        );


        ctx.fillStyle =
            "#67e8f9";

        ctx.fill();


        ctx.restore();

    }

}


// ============================================================
// DRAW ENEMIES
// ============================================================

function drawEnemies() {

    for (
        const enemy of enemies
    ) {

        ctx.save();


        ctx.shadowBlur = 18;

        ctx.shadowColor =
            "#ef4444";


        ctx.beginPath();

        ctx.arc(
            enemy.x,
            enemy.y,
            enemy.radius,
            0,
            Math.PI * 2
        );


        ctx.fillStyle =
            "#7f1d1d";

        ctx.fill();


        ctx.strokeStyle =
            "#fb7185";

        ctx.lineWidth = 2;

        ctx.stroke();


        // Eye

        ctx.beginPath();

        ctx.arc(
            enemy.x,
            enemy.y,
            4,
            0,
            Math.PI * 2
        );


        ctx.fillStyle =
            "#ff5577";

        ctx.fill();


        ctx.restore();

    }

}


// ============================================================
// DRAW HAZARDS
// ============================================================

function drawHazards() {

    for (
        const hazard of hazards
    ) {

        const alpha =
            Math.min(
                1,
                hazard.life
            );


        ctx.save();


        ctx.globalAlpha =
            alpha;


        ctx.beginPath();

        ctx.arc(
            hazard.x,
            hazard.y,
            hazard.radius,
            0,
            Math.PI * 2
        );


        ctx.fillStyle =
            "rgba(239,68,68,.12)";

        ctx.fill();


        ctx.strokeStyle =
            "#ef4444";

        ctx.lineWidth = 2;

        ctx.stroke();


        ctx.restore();

    }

}


// ============================================================
// DRAW PORTAL
// ============================================================

function drawPortals() {

    for (
        const portal of portals
    ) {

        portal.pulse +=
            0.04;


        const radius =
            portal.radius +
            Math.sin(
                portal.pulse
            ) *
            5;


        ctx.save();


        ctx.shadowBlur = 30;

        ctx.shadowColor =
            "#c084fc";


        ctx.beginPath();

        ctx.arc(
            portal.x,
            portal.y,
            radius,
            0,
            Math.PI * 2
        );


        ctx.strokeStyle =
            "#c084fc";

        ctx.lineWidth = 4;

        ctx.stroke();


        ctx.beginPath();

        ctx.arc(
            portal.x,
            portal.y,
            radius * .55,
            0,
            Math.PI * 2
        );


        ctx.strokeStyle =
            "#7c3aed";

        ctx.lineWidth = 2;

        ctx.stroke();


        ctx.restore();

    }

}


// ============================================================
// DRAW PARTICLES
// ============================================================

function drawParticles() {

    for (
        const p of particles
    ) {

        ctx.save();


        ctx.globalAlpha =
            Math.max(
                0,
                p.life
            );


        ctx.fillStyle =
            p.color;


        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            p.size,
            0,
            Math.PI * 2
        );


        ctx.fill();


        ctx.restore();

    }

}


// ============================================================
// DRAW EVERYTHING
// ============================================================

function draw() {

    drawBackground();

    drawHazards();

    drawPortals();

    drawCrystals();

    drawEnergy();

    drawEnemies();

    drawParticles();

    drawPlayer();

}


// ============================================================
// UPDATE
// ============================================================

function update(dt) {

    elapsed += dt;

    timeLeft -= dt;


    if (
        timeLeft <= 0
    ) {

        timeLeft = 0;

        endGame();

        return;

    }


    updatePlayer(dt);

    updateEnemies(dt);

    updateHazards(dt);

    collectObjects();

    updateDirectorSystem(dt);

    updateParticles(dt);


    // Increase difficulty with time

    level =
        Math.max(
            1,
            Math.floor(
                elapsed / 25
            ) + 1
        );


    updateHUD();

}


// ============================================================
// GAME LOOP
// ============================================================

function gameLoop(timestamp) {

    if (!gameRunning)
        return;


    const dt =
        Math.min(
            0.033,
            (
                timestamp -
                lastTime
            ) / 1000
        );


    lastTime =
        timestamp;


    if (!paused) {

        update(dt);

        draw();

    }


    requestAnimationFrame(
        gameLoop
    );

}


// ============================================================
// PAUSE
// ============================================================

function togglePause() {

    if (!gameRunning)
        return;


    paused =
        !paused;


    document
        .getElementById(
            "pauseScreen"
        )
        .classList.toggle(
            "hidden",
            !paused
        );

}


// ============================================================
// END GAME
// ============================================================

function endGame() {

    if (!gameRunning)
        return;


    gameRunning = false;


    document
        .getElementById(
            "gameOver"
        )
        .classList.remove(
            "hidden"
        );


    document
        .getElementById(
            "finalScore"
        )
        .textContent =
        Math.floor(score);


    document
        .getElementById(
            "finalMessage"
        )
        .textContent =

        shards >= 20

            ? "You escaped the collapsing dimension."

            : "The dimension consumed you.";

}


// ============================================================
// HUD
// ============================================================

function updateHUD() {

    document
        .getElementById(
            "shards"
        )
        .textContent =
        shards;


    document
        .getElementById(
            "score"
        )
        .textContent =
        Math.floor(score);


    document
        .getElementById(
            "timer"
        )
        .textContent =
        Math.ceil(
            timeLeft
        );


    document
        .getElementById(
            "energyBar"
        )
        .style.width =
        `${energy}%`;


    let hearts = "";

    for (
        let i = 0;
        i < 3;
        i++
    ) {

        hearts +=
            i < health
                ? "♥ "
                : "♡ ";

    }


    document
        .getElementById(
            "hearts"
        )
        .textContent =
        hearts;

}


// ============================================================
// NOTIFICATION
// ============================================================

let notificationTimer;


function showNotification(text) {

    const element =
        document.getElementById(
            "notification"
        );


    element.textContent =
        text;


    element.classList.add(
        "show"
    );


    clearTimeout(
        notificationTimer
    );


    notificationTimer =
        setTimeout(
            () => {

                element.classList.remove(
                    "show"
                );

            },
            1000
        );

}


// ============================================================
// BUTTONS
// ============================================================

document
    .getElementById(
        "startBtn"
    )
    .addEventListener(
        "click",
        startGame
    );


document
    .getElementById(
        "restartBtn"
    )
    .addEventListener(
        "click",
        () => {

            document
                .getElementById(
                    "gameOver"
                )
                .classList.add(
                    "hidden"
                );


            startGame();

        }
    );


// ============================================================
// INITIAL SCREEN
// ============================================================

draw();