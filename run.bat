@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul


call conda activate consisid_new

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
:: Configuration - Adjust these parameters as needed
set PYTHON_SCRIPT=tools/cache_inference/teacache_inference_consisid.py
set CKPTS_PATH=ckpts
set OUTPUT_PATH=result3

:: Image file - adjust this for each run if needed
set IMAGE_1="ConsisID-data\eval\face_images\1_stars_woman_Taylor_Swift_1.png"
set IMAGE_2="ConsisID-data\eval\face_images\117_ai_man_Huang_2.png"
set IMAGE_3="ConsisID-data\eval\face_images\124_ai_man_Yoshua_Bengio_4.png"

:: Prompt - adjust this for each run if needed
set PROMPT_1="A woman stands in a sunlit flower shop, her hands delicately adjusting stems within a half-arranged bouquet of blush-pink peonies while a stray petal clings to her floral-patterned apron; she gently brushes the pale fragment away with a practiced flick of her wrist. Leaning in until her nose nearly grazes a velvety soft-pink bloom, she deeply inhales its honeyed fragrance, her eyelids fluttering shut as a serene expression softens her gaze while the scent lingers warmly. Nearby, a vintage-style radio with a fabric grille emits the mellow, low-volume strains of an acoustic folk song, its melody weaving through the floral scents. Glistening water droplets bead and trickle like scattered diamonds across the weathered oak tabletop, left from freshly cut stems and overflowing vases. Among the peonies, clusters of vibrant red carnations in additional arrangements add bold contrasting color to the sun-dappled scene"

set PROMPT_2="The video shows a man deeply focused while working in a cozy, intimate café, leaning over a weathered leather-bound notebook on a wooden surface. He twists a sleek silver pen between his fingers, frozen mid-thought as he stares at a half-written sentence. After a moment, he pauses to lift a steaming mug of latte, blowing gently across its surface—a visible plume of steam curls upward, momentarily fogging the thin wire frames of his glasses. Behind him, rustic wooden shelves line the background, meticulously stocked with glass jars overflowing with assorted homemade cookies, creating a bakery-style display. Soft ambient jazz melodies drift faintly through the space, though no visible source like a record player is shown, enhancing the tranquil atmosphere. Warm, diffused lighting casts gentle shadows, emphasizing the café’s inviting ambiance."
set PROMPT_3="The video captures a man repairing a vintage bicycle with a classic diamond frame, wire-spoked wheels, and a worn leather saddle in a narrow, gritty alley characterized by peeling paint, rust stains, and scattered industrial debris. His sleeves are rolled up to reveal a faded anchor tattoo on his weathered forearm; he deliberately wipes thick, black oil from his hands using a grimy, oil-stained rag before carefully tightening a bolt with a wrench. Sunlight slants through a tangle of overhead electrical wires, projecting sharp, lace-like shadows across the cobblestones and a cluster of scattered tools—including a crescent wrench, screwdrivers, and a metal toolbox—at his feet. A mottled tabby cat with alert green eyes curls motionlessly on the bike’s rear rack, intently observing the man’s precise hand movements without flinching at the metallic sounds, its stillness suggesting a stray accustomed to human presence in the industrial setting."
set PROMPT_4="A woman stands in a sunlit flower shop, her hands delicately adjusting stems within a half-arranged bouquet of blush-pink peonies while a stray petal clings to her floral-patterned apron; she gently brushes the pale fragment away with a practiced flick of her wrist. Leaning in until her nose nearly grazes a velvety soft-pink bloom, she deeply inhales its honeyed fragrance, her eyelids fluttering shut as a serene expression softens her gaze while the scent lingers warmly. Nearby, a vintage-style radio with a fabric grille emits the mellow, low-volume strains of an acoustic folk song, its melody weaving through the floral scents. Glistening water droplets bead and trickle like scattered diamonds across the weathered oak tabletop, left from freshly cut stems and overflowing vases. Among the peonies, clusters of vibrant red carnations in additional arrangements add bold contrasting color to the sun-dappled scene"
set PROMPT_5="The video depicts a man in his early thirties with tousled chestnut hair and thin wire-rimmed glasses, deeply engrossed in work within a cramped, warmly lit café. He leans intently over a thick, dark brown leather-bound notebook with frayed edges, its yellowed pages filled with dense blue-ink script. His right hand absentmindedly twists a slender, polished silver fountain pen between calloused fingers, knuckles whitening slightly as he stares at a half-completed sentence mid-page, brows faintly furrowed in concentration. After a prolonged pause, he lifts a chipped white ceramic latte mug—its surface still swirling with condensed steam from freshly poured coffee—and gently blows across the velvety foam. Delicate tendrils of vapor spiral upward in thick undulating curls, instantly condensing into droplets that fog his glasses, forcing him to push them higher on his nose. Behind him, a weathered dark oak counter is piled haphazardly with clear glass apothecary jars stuffed with chunky oatmeal-raisin and chocolate-chunk cookies, their dented metal lids bearing handwritten chalk labels. Soft, muted 1950s jazz—featuring a brushed-snare rhythm and a smoky tenor saxophone—plays from a vintage record player tucked beside a potted fern, blending seamlessly with the low clatter of distant cutlery and the café’s intimate hum."
set PROMPT_6="The video captures a middle-aged man meticulously restoring a weathered 1950s roadster bicycle in a narrow, cobblestone alley flanked by decaying brick buildings. His rolled-up chambray shirt reveals a faded anchor tattoo on his sun-freckled forearm as he pauses to wipe thick grease from his fingers with a grimy cotton rag before tightening a rear wheel bolt with a crescent wrench. Sunlight filters through a tangle of overhead electrical wires, etching elongated, parallel shadows across rust-speckled tools scattered on the cobblestones at his feet—a worn adjustable wrench, a glass jar of assorted bolts, and a folded maintenance manual. Positioned safely on a stable cobblestone beside the rear wheel, a ginger tabby cat observes him with half-closed eyes, its tail occasionally flicking, maintaining visual connection while avoiding moving parts. The alley’s atmosphere is anchored by textured details: peeling vintage advertisements on damp-stained brick walls, a faint sheen of oil slicks in rainwater puddles, and oxidized metal drainpipes running vertically beside him. Dust motes dance in the slanted light beams, while the faint metallic tang of lubricant hangs in the air, mingling with visible grime streaking the bicycle’s chrome frame and oil stains blooming across the man’s work trousers."
set PROMPT_7="A woman stands in a sunlit flower shop, carefully selecting individual peony stems from a work counter cluttered with floral shears, damp paper cones, and scattered green foliage clippings. She lifts a single blush-pink bloom closer to eye level, her fingers gently brushing a stray petal from her striped apron before leaning in to inhale its delicate fragrance—her gaze softening momentarily. Behind her, vibrant red carnations occupy a raised shelf near the window, safely distant from the active workspace. A compact waterproof Bluetooth speaker sits on a dry ledge away from water sources, playing a muted folk melody. Sunlight catches on water droplets beading along glass vases, pooling in faint stains on the weathered wooden tabletop where loose petals and soil smudges mark the busy work zone. As she adjusts a stem’s angle, she subtly shifts her weight onto one hip, tucks a loose strand of hair behind her ear, and pauses to hum softly to the music before resuming her task. The scene balances poetic stillness with the gentle chaos of commerce—neatly arranged customer bouquets line the background shelves, yet the work counter hums with focused, slightly messy creativity."
set PROMPT_8="The video depicts a middle-aged man with silver-streaked hair wearing a faded cable-knit sweater, seated in a dimly lit mahogany-paneled study lined with floor-to-ceiling bookshelves filled with leather-bound volumes. He carefully flips through a weathered stack of sepia-toned family photographs with slightly curled edges, pausing at one showing his younger self—a freckle-faced boy around eight years old—wearing a vintage striped soccer jersey and grass-stained shorts, beaming while clutching a scuffed leather soccer ball under one arm. His weathered thumb traces the photograph’s frayed, embossed border with deliberate tenderness, his gaze distant and wistful. A brass desk lamp with a green glass shade casts a focused pool of warm amber light across his furrowed brow and the photograph, deepening the shadows in the room. Beside him on the polished hardwood floor rests a chipped porcelain teacup half-filled with cooled Earl Grey tea, its surface glazed with a faint sheen, while a precarious tower of antique hardcover classics leans against the desk leg, their gilded spines barely visible."
set PROMPT_9="The video depicts a middle-aged street musician with weathered hands and subtle creases around his eyes, deeply engrossed in playing a sun-faded acoustic guitar at a bustling urban intersection during twilight's amber-to-indigo transition. His calloused fingers execute rapid flamenco-style flourishes across the steel strings, producing intricate melodies that mingle with the low hum of evening traffic. Three distinct passersby—a young couple clasping hands, an elderly woman clutching a fabric shopping bag, and a teenager in a graphic hoodie—simultaneously pause, their silhouettes softened by the fading light, as they discreetly drop crumpled bills and coins (including a silver euro and copper pennies) into the guitar case lined with faded band stickers. Behind him, pulsating neon signs from a retro diner (glowing aquamarine) and a vintage record store (vibrant crimson) cast dynamic reflections that fracture across the rain-slicked pavement, their light rippling like liquid fire over his scuffed, knee-high leather boots caked with urban grime. As a small child in a red puffy jacket skips past while clutching a parent's hand, the musician momentarily pauses his strumming, his sun-weathered face crinkling into a weary but genuine smile as he dips his chin in a slow, appreciative nod; the child beams, waving both hands enthusiastically before being gently guided away by the parent, their figures dissolving into the dusk-lit crowd."
set PROMPT_10="A woman walks along a gently sloping beach at sunset, her bare feet leaving faint, transient imprints in the warm, damp sand near the water’s edge—impressions softened and partially refilled by each receding wave. She carries a woven seagrass basket balanced carefully on one hip, its contents shifting softly with her movements: half a dozen iridescent scallop and cockle shells already nestled inside, glinting faintly in the dwindling light. Occasionally, she pauses to bend at the knees (not the waist), plucking a particularly polished moon snail or conical turban shell from the damp sand where the foam retreats, tucking it gently among the others in her basket with a small, satisfied smile. The rhythmic waves surge close enough to brush her toes when she lingers, prompting her to instinctively shift her path—a subtle sidestep or angled retreat onto drier sand—to avoid the frothy advance. Behind her, the sky melts into watercolor hues of tangerine and rose, catching in her windswept hair as smaller waves collapse onto the shore with a muffled hiss. Farther inland, beyond the reach of constant waves, sparse tufts of hardy beach grass anchor low dunes, their silhouettes soft against the fading light."
set PROMPT_11="The video captures a middle-aged librarian with silver-streaked hair and round spectacles, dressed in a faded tweed vest over a collared shirt, kneeling on a worn Persian rug beside a towering mahogany bookshelf in a grand, wood-paneled library. He stretches upward with deliberate slowness, fingers brushing against the cracked spine of an oversized leather-bound volume perched on the uppermost shelf, its gilded title barely visible beneath layers of aged dust. With careful precision, he grips the book’s edges using both hands to avoid dislodging neighboring volumes, easing it outward until the aged leather groans softly in protest. Once freed, he cradles the heavy tome against his chest and uses his open palm to gently swipe away accumulated grime in three smooth strokes, sending a faint cloud of fine particles drifting into a sunbeam. After pausing to blow lightly on the revealed embossed patterns, he cracks open the brittle pages with reverence, thumbing past yellowed folios until his fingers pause at the silk-bookmark-marked index section. The library envelops him in profound silence, punctuated only by the distant whisper of turning pages from shadowed reading nooks and the faint ticking of a grandfather clock. Late afternoon sunlight pours through towering arched windows, slicing through the dimness in golden columns that catch and suspend swirling dust motes in luminous, slow-motion beams while casting elongated shadows across polished oak tables."
set PROMPT_12="The video depicts a middle-aged man in a slightly worn navy hoodie and jeans standing inside a dimly lit 24-hour convenience store during late night, clutching a steaming microwaveable bowl of beef-flavoramen sealed with a vented plastic lid. He leans wearily against a chipped laminate countertop, shoulders slumped, eyes intensely fixed on a countertop microwave’s glowing red digital timer as it counts down from under 1 minute. To combat the pervasive chill, he alternates between vigorously rubbing his gloveless, red-knuckled hands together—producing faint friction sounds—and exhaling sharp breaths that condense into wispy, ephemeral clouds dissipating instantly under the harsh fluorescent glare. Behind the register, a young clerk with tousled hair and a disinterested expression, dressed in a crisp store-issued polo shirt, idly flips through a glossy celebrity magazine, its pages rustling softly as his gaze drifts between the pages and the man. The store’s environment is defined by twin ceiling-mounted fluorescent tubes emitting a sterile, buzzing hum that vibrates faintly in the quiet space, casting sharp reflections on polished linoleum floors while illuminating neatly aligned snack racks, a half-stocked beverage cooler, and a dormant coffee machine. Through the glass entrance door, a deserted dark street is faintly visible, underscoring the isolation of the scene."
set PROMPT_13="A woman stands in a warm, cluttered art studio filled with weathered oak easels stacked against terracotta-tiled walls, her calloused fingers deftly swirling creamy cerulean and cobalt pigments into a mosaic of vibrant hues on a splintered wooden palette. She dips a frayed sable-hair brush into a viscous dollop of luminous sky-blue paint, the bristles glistening as she pauses, tilting her head slightly to squint critically at a linen canvas propped on a sturdy easel—where a half-finished landscape reveals charcoal-sketched rolling hills beneath a hazily outlined sun, its rays barely suggested with faint yellow washes. With deliberate, slightly tremulous strokes, she drags the brush diagonally across the canvas upper quadrant, layering translucent washes of blue that deepen toward the horizon line, the wet pigment catching the light. Dried smudges of ochre, burnt sienna, and emerald green speckle her faded indigo jeans like abstract constellations, while above the easel, a large arched window frames slanted beams of late-afternoon sunlight that illuminate dancing dust motes and make the iridescent silver-gold glitter woven through her loose chestnut braids shimmer like captured stardust."
set PROMPT_14="The video depicts a muscular man in his late 20s, clad in a sweat-drenched gray sleeveless tank top and black moisture-wicking shorts, meticulously sanitizing a commercial-grade treadmill after an intense workout. He grips a lemon-scented disinfectant wipe with deliberate focus, applying firm pressure in overlapping circular motions across the sweat-streaked handlebars, digital console buttons, and textured running belt surface—his biceps visibly straining and flexing with defined striations as he contorts to reach the uppermost edge of the machine’s frame. Beside him on the speckled rubber gym floor rests a translucent 24oz plastic water bottle, its surface beaded with condensation that pools slightly beneath it; a partially detached label curls upward at the corner, exposing sticky adhesive residue beneath a faded energy drink logo. Overhead, the gym’s ceiling-mounted speakers emit a current Top 40 pop track with pulsating synth beats and crisp electronic undertones, its volume subtly softened by the rhythmic metallic clatter of distant Olympic barbell plates colliding during deadlift sets in the adjacent weightlifting zone."
set PROMPT_15="The video captures a middle-aged man in a faded denim work shirt and round spectacles standing precariously on the second rung of a narrow wooden A-frame step-ladder, his fingers brushing aged dust from the spine of a thick leather-bound volume before carefully extracting it from the topmost shelf of a towering mahogany bookcase that scrapes the ceiling. He grips the worn crimson-covered book with both hands as the ladder wobbles slightly under his weight, then descends cautiously to stand on the floor, holding the book open at chest height. Leaning into a sunbeam piercing through a central glass skylight, he gently flips past the frontispiece to reveal a yellowed flyleaf with a faded cursive dedication, tracing the words with a calloused fingertip while his lips move silently in focused reading. From behind a cluttered mahogany front desk piled with ledgers and a brass bell, a kind-faced store employee with silver hair tied in a bun raises her right hand in a warm, unhurried wave, her entire figure visible as she smiles—later asking, Finding something special? in a soft voice. Dust motes dance in slanted columns of afternoon light from the skylight, illuminating swirling particles that catch the light like glitter, while the air hangs thick with the scent of aged paper, cedar shelves, and a vanilla-scented candle melting near the cash register. The bookstore interior features floor-to-ceiling dark oak shelves groaning under decades-old leather volumes, antique brass reading lamps casting warm pools on faded Persian rugs, and framed botanical prints adorning the walls—every detail reinforcing timeless intimacy. After lingering over the dedication with a nostalgic expression, the man gently closes the book, his thumb resting thoughtfully on the embossed title before carefully reshelving it at eye level rather than returning it to the unreachable top shelf. The skylight’s view reveals only a sliver of weathered brickwork and ivy-strewn walls outside, deliberately obscuring any modern construction, while the employee gives another gentle nod as he steps away from the ladder."

set REL_L1_THRESH=0.1

set NUM_STEPS=50

echo Starting 3 runs of the Python script...
echo.

:: Run 1
echo ===== RUN 1 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_1% --prompt %PROMPT_1% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 1
    pause
    exit /b 1
)
echo.


:: Run 2
echo ===== RUN 2 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_2% --prompt %PROMPT_2% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 2
    pause
    exit /b 1
)
echo.

:: Run 3
echo ===== RUN 3 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_3% --prompt %PROMPT_3% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 3
    pause
    exit /b 1
)
echo.
goto end
:: Run 4
echo ===== RUN 4 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_1% --prompt %PROMPT_4% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 4
    pause
    exit /b 1
)
echo.

:: Run 5
echo ===== RUN 5 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_2% --prompt %PROMPT_5% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 5
    pause
    exit /b 1
)
echo.

:: Run 6
echo ===== RUN 6 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_3% --prompt %PROMPT_6% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 6
    pause
    exit /b 1
)
echo.

:: Run 7
echo ===== RUN 7 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_1% --prompt %PROMPT_7% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 7
    pause
    exit /b 1
)
echo.

:: Run 8
echo ===== RUN 8 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_2% --prompt %PROMPT_8% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 8
    pause
    exit /b 1
)
echo.

:: Run 9
echo ===== RUN 9 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_3% --prompt %PROMPT_9% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 9
    pause
    exit /b 1
)
echo.

:: Run 10
echo ===== RUN 10 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_1% --prompt %PROMPT_10% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 10
    pause
    exit /b 1
)
echo.

:: Run 11
echo ===== RUN 11 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_2% --prompt %PROMPT_11% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 11
    pause
    exit /b 1
)
echo.

:: Run 12
echo ===== RUN 12 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_3% --prompt %PROMPT_12% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 12
    pause
    exit /b 1
)
echo.

:: Run 13
echo ===== RUN 13 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_1% --prompt %PROMPT_13% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 13
    pause
    exit /b 1
)
echo.

:: Run 14
echo ===== RUN 14 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_2% --prompt %PROMPT_14% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 14
    pause
    exit /b 1
)
echo.

:: Run 15
echo ===== RUN 15 =====
python %PYTHON_SCRIPT% --rel_l1_thresh %REL_L1_THRESH% --ckpts_path %CKPTS_PATH% --image %IMAGE_3% --prompt %PROMPT_15% --output_path %OUTPUT_PATH% --num_infer_steps %NUM_STEPS%
if errorlevel 1 (
    echo Error in Run 15
    pause
    exit /b 1
)
echo.

echo All 15 runs completed successfully!
:end

pause
