import { Link } from 'react-router-dom';

interface CosmeticLinkProps {
    to: string;
    children: React.ReactNode;
    target?: string;
}

export default function CosmeticLink({ to, children, target }: CosmeticLinkProps) {
    return (
        <Link
            to={to}
            target={target}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '7px 12px',
                background: 'rgba(13, 13, 20, 0.88)',
                border: '1px solid rgba(255,255,255,0.09)',
                borderRadius: 10,
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
                color: 'rgba(255,255,255,0.75)',
                fontSize: 11,
                fontWeight: 600,
                textDecoration: 'none',
            }}
        >
            {children}
        </Link>
    );
}